#!/usr/bin/env python3
"""Parse NSight Systems exports (sqlite/csv/txt) into normalized tables."""

from __future__ import annotations

import argparse
import json
import re
import sqlite3
import subprocess
import tempfile
from collections import defaultdict
from pathlib import Path
from typing import Any, Dict, List

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd


def _pick_input_file(trace_dir: Path) -> Path | None:
    patterns = ("*.sqlite", "*.csv", "*.txt", "*.nsys-rep")
    candidates: List[Path] = []
    for p in patterns:
        candidates.extend(trace_dir.rglob(p))
    candidates = [p for p in candidates if p.is_file()]
    if not candidates:
        return None

    def sqlite_valid(path: Path) -> bool:
        try:
            conn = sqlite3.connect(str(path))
            try:
                conn.execute("SELECT name FROM sqlite_master LIMIT 1").fetchall()
            finally:
                conn.close()
            return True
        except Exception:
            return False

    # Prefer newest valid sqlite first.
    valid_sqlite = [p for p in candidates if p.suffix == ".sqlite" and sqlite_valid(p)]
    if valid_sqlite:
        return sorted(valid_sqlite, key=lambda p: p.stat().st_mtime, reverse=True)[0]

    # Then fall back to newest nsys-rep/csv/txt.
    fallback = [p for p in candidates if p.suffix != ".sqlite"]
    if fallback:
        return sorted(fallback, key=lambda p: p.stat().st_mtime, reverse=True)[0]

    # Only invalid sqlite exists.
    return sorted(candidates, key=lambda p: p.stat().st_mtime, reverse=True)[0]


def _table_names(conn: sqlite3.Connection) -> List[str]:
    rows = conn.execute("SELECT name FROM sqlite_master WHERE type='table'").fetchall()
    return [r[0] for r in rows]


def _table_columns(conn: sqlite3.Connection, table: str) -> List[str]:
    rows = conn.execute(f"PRAGMA table_info('{table}')").fetchall()
    return [r[1] for r in rows]


def _extract_kernels_from_sqlite(conn: sqlite3.Connection) -> List[Dict[str, Any]]:
    out: List[Dict[str, Any]] = []
    tables = set(_table_names(conn))
    kernel_table = None
    for t in ("CUPTI_ACTIVITY_KIND_KERNEL", "CUDA_GPU_KERNEL"):
        if t in tables:
            kernel_table = t
            break
    if kernel_table is None:
        return out

    cols = _table_columns(conn, kernel_table)
    name_col = "demangledName" if "demangledName" in cols else ("shortName" if "shortName" in cols else ("name" if "name" in cols else None))
    start_col = "start" if "start" in cols else ("start_time" if "start_time" in cols else None)
    end_col = "end" if "end" in cols else ("end_time" if "end_time" in cols else None)
    stream_col = "streamId" if "streamId" in cols else ("stream" if "stream" in cols else None)
    device_col = "deviceId" if "deviceId" in cols else ("device" if "device" in cols else None)

    select_cols = [c for c in [name_col, start_col, end_col, stream_col, device_col] if c]
    if not select_cols:
        return out

    df = pd.read_sql_query(f"SELECT {', '.join(select_cols)} FROM {kernel_table}", conn)
    for _, row in df.iterrows():
        start_ns = int(row[start_col]) if start_col and pd.notna(row[start_col]) else 0
        end_ns = int(row[end_col]) if end_col and pd.notna(row[end_col]) else start_ns
        out.append(
            {
                "name": str(row[name_col]) if name_col and pd.notna(row[name_col]) else "Unknown",
                "start_time_ns": start_ns,
                "duration_ns": max(0, end_ns - start_ns),
                "grid_size": "",
                "block_size": "",
                "registers": 0,
                "shared_memory": 0,
                "device": int(row[device_col]) if device_col and pd.notna(row[device_col]) else 0,
                "stream": int(row[stream_col]) if stream_col and pd.notna(row[stream_col]) else 0,
            }
        )
    return out


def _extract_memcpy_from_sqlite(conn: sqlite3.Connection) -> List[Dict[str, Any]]:
    out: List[Dict[str, Any]] = []
    tables = set(_table_names(conn))
    memcpy_table = None
    for t in ("CUPTI_ACTIVITY_KIND_MEMCPY", "CUDA_GPU_MEMCPY"):
        if t in tables:
            memcpy_table = t
            break
    if memcpy_table is None:
        return out

    cols = _table_columns(conn, memcpy_table)
    start_col = "start" if "start" in cols else ("start_time" if "start_time" in cols else None)
    end_col = "end" if "end" in cols else ("end_time" if "end_time" in cols else None)
    bytes_col = "bytes" if "bytes" in cols else ("size" if "size" in cols else None)
    copy_col = "copyKind" if "copyKind" in cols else ("kind" if "kind" in cols else None)
    src_col = "srcKind" if "srcKind" in cols else None
    dst_col = "dstKind" if "dstKind" in cols else None

    select_cols = [c for c in [start_col, end_col, bytes_col, copy_col, src_col, dst_col] if c]
    if not select_cols:
        return out

    df = pd.read_sql_query(f"SELECT {', '.join(select_cols)} FROM {memcpy_table}", conn)
    for _, row in df.iterrows():
        start_ns = int(row[start_col]) if start_col and pd.notna(row[start_col]) else 0
        end_ns = int(row[end_col]) if end_col and pd.notna(row[end_col]) else start_ns
        size = int(row[bytes_col]) if bytes_col and pd.notna(row[bytes_col]) else 0
        duration_ns = max(1, end_ns - start_ns)
        throughput_gbs = (size / duration_ns) if duration_ns > 0 else 0.0  # bytes/ns == GB/s
        out.append(
            {
                "operation": str(row[copy_col]) if copy_col and pd.notna(row[copy_col]) else "memcpy",
                "start_time_ns": start_ns,
                "duration_ns": duration_ns,
                "size": size,
                "throughput": float(throughput_gbs),
                "source": str(row[src_col]) if src_col and pd.notna(row[src_col]) else "unknown",
                "destination": str(row[dst_col]) if dst_col and pd.notna(row[dst_col]) else "unknown",
            }
        )
    return out


def _extract_runtime_from_sqlite(conn: sqlite3.Connection) -> List[Dict[str, Any]]:
    out: List[Dict[str, Any]] = []
    tables = set(_table_names(conn))
    runtime_table = None
    for t in ("CUPTI_ACTIVITY_KIND_RUNTIME", "CUDA_RUNTIME_API"):
        if t in tables:
            runtime_table = t
            break
    if runtime_table is None:
        return out

    cols = _table_columns(conn, runtime_table)
    start_col = "start" if "start" in cols else ("start_time" if "start_time" in cols else None)
    end_col = "end" if "end" in cols else ("end_time" if "end_time" in cols else None)
    name_col = "name" if "name" in cols else ("apiName" if "apiName" in cols else None)
    pid_col = "globalPid" if "globalPid" in cols else ("processId" if "processId" in cols else None)
    tid_col = "threadId" if "threadId" in cols else None

    select_cols = [c for c in [name_col, start_col, end_col, pid_col, tid_col] if c]
    if not select_cols:
        return out

    df = pd.read_sql_query(f"SELECT {', '.join(select_cols)} FROM {runtime_table}", conn)
    for _, row in df.iterrows():
        start_ns = int(row[start_col]) if start_col and pd.notna(row[start_col]) else 0
        end_ns = int(row[end_col]) if end_col and pd.notna(row[end_col]) else start_ns
        out.append(
            {
                "name": str(row[name_col]) if name_col and pd.notna(row[name_col]) else "runtime_api",
                "start_time_ns": start_ns,
                "duration_ns": max(0, end_ns - start_ns),
                "process_id": int(row[pid_col]) if pid_col and pd.notna(row[pid_col]) else 0,
                "thread_id": int(row[tid_col]) if tid_col and pd.notna(row[tid_col]) else 0,
                "result": "unknown",
            }
        )
    return out


def parse_sqlite(path: Path) -> Dict[str, Any]:
    try:
        conn = sqlite3.connect(str(path))
        try:
            parsed = {
                "gpu_kernels": _extract_kernels_from_sqlite(conn),
                "memory_operations": _extract_memcpy_from_sqlite(conn),
                "cuda_api_calls": _extract_runtime_from_sqlite(conn),
            }
        finally:
            conn.close()
        return parsed
    except sqlite3.DatabaseError as e:
        raise RuntimeError(f"{path} is not a readable sqlite DB: {e}") from e


def _export_nsys_rep_to_sqlite(nsys_rep: Path) -> Path:
    out_prefix = Path(tempfile.mkdtemp(prefix="nsys_export_")) / nsys_rep.stem
    cmd = [
        "nsys",
        "export",
        "--type",
        "sqlite",
        "--output",
        str(out_prefix),
        str(nsys_rep),
    ]
    subprocess.run(cmd, check=True)
    sqlite_path = out_prefix.with_suffix(".sqlite")
    if not sqlite_path.exists():
        raise RuntimeError(f"nsys export succeeded but sqlite not found: {sqlite_path}")
    return sqlite_path


def parse_csv(path: Path) -> Dict[str, Any]:
    df = pd.read_csv(path)
    cols = [str(c).lower() for c in df.columns]
    parsed = {"gpu_kernels": [], "memory_operations": [], "cuda_api_calls": []}

    if "kernel" in " ".join(cols):
        for _, row in df.iterrows():
            parsed["gpu_kernels"].append(
                {
                    "name": row.get("Kernel Name", row.get("Name", "Unknown")),
                    "start_time_ns": row.get("Start (ns)", 0),
                    "duration_ns": row.get("Duration (ns)", 0),
                    "grid_size": row.get("Grid Size", ""),
                    "block_size": row.get("Block Size", ""),
                    "registers": row.get("Registers Per Thread", 0),
                    "shared_memory": row.get("Shared Memory Configuration Size", 0),
                    "device": row.get("Device", 0),
                    "stream": row.get("Stream", 0),
                }
            )
    elif "memcpy" in " ".join(cols) or "memory" in " ".join(cols):
        for _, row in df.iterrows():
            parsed["memory_operations"].append(
                {
                    "operation": row.get("Operation", row.get("Name", "Unknown")),
                    "start_time_ns": row.get("Start (ns)", 0),
                    "duration_ns": row.get("Duration (ns)", 0),
                    "size": row.get("Size", 0),
                    "throughput": row.get("Throughput", 0),
                    "source": row.get("Source Device", "Unknown"),
                    "destination": row.get("Destination Device", "Unknown"),
                }
            )
    else:
        for _, row in df.iterrows():
            parsed["cuda_api_calls"].append(
                {
                    "name": row.get("API Name", row.get("Name", "Unknown")),
                    "start_time_ns": row.get("Start (ns)", 0),
                    "duration_ns": row.get("Duration (ns)", 0),
                    "process_id": row.get("Process ID", 0),
                    "thread_id": row.get("Thread ID", 0),
                    "result": row.get("Result", "Unknown"),
                }
            )
    return parsed


def parse_txt(path: Path) -> Dict[str, Any]:
    content = path.read_text(encoding="utf-8", errors="ignore")
    kernels = []
    for m in re.finditer(r"Kernel:\s*(.*?)\s*Duration:\s*([\d.]+)\s*ms", content, re.IGNORECASE):
        kernels.append(
            {
                "name": m.group(1).strip(),
                "start_time_ns": 0,
                "duration_ns": int(float(m.group(2)) * 1e6),
                "grid_size": "",
                "block_size": "",
                "registers": 0,
                "shared_memory": 0,
                "device": 0,
                "stream": 0,
            }
        )
    return {"gpu_kernels": kernels, "memory_operations": [], "cuda_api_calls": []}


def generate_summary(parsed: Dict[str, Any]) -> Dict[str, Any]:
    kernels = parsed["gpu_kernels"]
    memops = parsed["memory_operations"]
    apis = parsed["cuda_api_calls"]
    summary = {
        "total_kernels": len(kernels),
        "total_memory_ops": len(memops),
        "total_api_calls": len(apis),
        "total_gpu_time_ns": float(sum(float(k.get("duration_ns", 0)) for k in kernels)),
        "total_memory_transfer_time_ns": float(sum(float(m.get("duration_ns", 0)) for m in memops)),
        "kernel_stats": {},
        "memory_stats": {},
        "api_stats": {},
    }

    kernel_groups: Dict[str, List[float]] = defaultdict(list)
    for k in kernels:
        kernel_groups[str(k.get("name", "Unknown"))].append(float(k.get("duration_ns", 0)))
    for name, durs in kernel_groups.items():
        summary["kernel_stats"][name] = {
            "count": len(durs),
            "total_duration_ns": float(sum(durs)),
            "avg_duration_ns": float(np.mean(durs)),
            "max_duration_ns": float(max(durs)),
            "min_duration_ns": float(min(durs)),
        }

    mem_groups: Dict[str, List[Dict[str, Any]]] = defaultdict(list)
    for m in memops:
        mem_groups[str(m.get("operation", "Unknown"))].append(m)
    for op, rows in mem_groups.items():
        durs = [float(x.get("duration_ns", 0)) for x in rows]
        summary["memory_stats"][op] = {
            "count": len(rows),
            "total_duration_ns": float(sum(durs)),
            "avg_duration_ns": float(np.mean(durs) if durs else 0.0),
            "total_size": int(sum(int(x.get("size", 0) or 0) for x in rows)),
        }

    api_groups: Dict[str, List[float]] = defaultdict(list)
    for a in apis:
        api_groups[str(a.get("name", "Unknown"))].append(float(a.get("duration_ns", 0)))
    for name, durs in api_groups.items():
        summary["api_stats"][name] = {
            "count": len(durs),
            "total_duration_ns": float(sum(durs)),
            "avg_duration_ns": float(np.mean(durs)),
        }
    return summary


def save_outputs(parsed: Dict[str, Any], source: Path, output_dir: Path) -> None:
    output_dir.mkdir(parents=True, exist_ok=True)

    if parsed["gpu_kernels"]:
        pd.DataFrame(parsed["gpu_kernels"]).to_csv(output_dir / "gpu_kernels.csv", index=False)
    if parsed["memory_operations"]:
        pd.DataFrame(parsed["memory_operations"]).to_csv(output_dir / "memory_operations.csv", index=False)
    if parsed["cuda_api_calls"]:
        pd.DataFrame(parsed["cuda_api_calls"]).to_csv(output_dir / "cuda_api_calls.csv", index=False)

    summary = generate_summary(parsed)
    payload = {"source_trace": str(source), "summary": summary}
    (output_dir / "nsys_summary.json").write_text(json.dumps(payload, indent=2, ensure_ascii=False), encoding="utf-8")

    with open(output_dir / "nsys_summary.txt", "w", encoding="utf-8") as f:
        f.write("NSight Systems Summary\n")
        f.write("=" * 24 + "\n")
        f.write(f"Source: {source}\n")
        f.write(f"Kernels: {summary['total_kernels']}\n")
        f.write(f"Memcpy ops: {summary['total_memory_ops']}\n")
        f.write(f"CUDA API calls: {summary['total_api_calls']}\n")
        f.write(f"GPU time: {summary['total_gpu_time_ns']/1e6:.3f} ms\n")


def create_figure(parsed: Dict[str, Any], fig_dir: Path) -> None:
    fig_dir.mkdir(parents=True, exist_ok=True)
    kernels = parsed["gpu_kernels"]
    if not kernels:
        return
    df = pd.DataFrame(kernels)
    if "duration_ns" not in df.columns:
        return
    top = df.groupby("name", as_index=False)["duration_ns"].sum().sort_values("duration_ns", ascending=False).head(15)
    plt.figure(figsize=(11, 5))
    plt.bar(range(len(top)), top["duration_ns"] / 1e6)
    plt.xticks(range(len(top)), [str(x)[:28] for x in top["name"]], rotation=35, ha="right")
    plt.ylabel("Total duration (ms)")
    plt.title("Top NSYS Kernels by Total Duration")
    plt.tight_layout()
    plt.savefig(fig_dir / "nsys_top_kernels.png", dpi=220, bbox_inches="tight")
    plt.close()


def main() -> None:
    parser = argparse.ArgumentParser(description="Parse NSYS outputs (sqlite/csv/txt)")
    parser.add_argument("--trace-dir", default="results/traces", help="Directory containing nsys exports")
    parser.add_argument("--trace-file", default=None, help="Optional explicit file")
    parser.add_argument("--output-dir", default="results/analysis/profiling/nsys", help="Output table directory")
    parser.add_argument("--fig-dir", default="results/figures/profiling/nsys", help="Output figure directory")
    args = parser.parse_args()

    source = Path(args.trace_file) if args.trace_file else _pick_input_file(Path(args.trace_dir))
    if source is None or not source.exists():
        raise FileNotFoundError(f"No NSYS input found (trace_dir={args.trace_dir}, trace_file={args.trace_file})")
    if source.suffix == ".nsys-rep":
        sqlite_source = _export_nsys_rep_to_sqlite(source)
        parsed = parse_sqlite(sqlite_source)
    elif source.suffix == ".sqlite":
        parsed = parse_sqlite(source)
    elif source.suffix == ".csv":
        parsed = parse_csv(source)
    else:
        parsed = parse_txt(source)

    save_outputs(parsed, source, Path(args.output_dir))
    create_figure(parsed, Path(args.fig_dir))
    print(f"[nsys parse] done. outputs: {args.output_dir}")


if __name__ == "__main__":
    main()
