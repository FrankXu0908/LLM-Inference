#!/usr/bin/env python3
"""Parse PyTorch profiler trace JSON into structured tables and quick plots."""

from __future__ import annotations

import argparse
import json
from collections import defaultdict
from pathlib import Path
from typing import Any, Dict, List

import matplotlib.pyplot as plt
import pandas as pd


def _pick_trace_file(trace_dir: Path) -> Path | None:
    candidates = []
    for pattern in ("*.trace.json", "*.json"):
        candidates.extend(trace_dir.rglob(pattern))
    candidates = [p for p in candidates if p.is_file() and "summary" not in p.name]
    if not candidates:
        return None
    return sorted(candidates, key=lambda p: p.stat().st_mtime, reverse=True)[0]


def parse_metadata(trace_data: Dict[str, Any]) -> Dict[str, Any]:
    events = trace_data.get("traceEvents", [])
    timestamps = [evt.get("ts") for evt in events if isinstance(evt.get("ts"), (int, float))]
    if timestamps:
        start_ts = min(timestamps)
        end_ts = max(timestamps)
    else:
        start_ts = None
        end_ts = None

    return {
        "version": trace_data.get("version", "unknown"),
        "start_time_us": start_ts,
        "end_time_us": end_ts,
        "total_duration_us": (end_ts - start_ts) if start_ts is not None and end_ts is not None else 0,
    }


def parse_operators(trace_data: Dict[str, Any]) -> List[Dict[str, Any]]:
    out: List[Dict[str, Any]] = []
    for evt in trace_data.get("traceEvents", []):
        if evt.get("ph") != "X" or "name" not in evt:
            continue
        args = evt.get("args", {}) if isinstance(evt.get("args"), dict) else {}
        out.append(
            {
                "name": evt.get("name", ""),
                "category": evt.get("cat", "unknown"),
                "start_time_us": evt.get("ts", 0),
                "duration_us": evt.get("dur", 0),
                "process_id": evt.get("pid", 0),
                "thread_id": evt.get("tid", 0),
                "input_shapes": json.dumps(args.get("Input Dims", args.get("input_shapes", []))),
                "output_shapes": json.dumps(args.get("output_shapes", [])),
                "flops": args.get("flops", 0),
                "memory": args.get("memory", 0),
            }
        )
    return out


def parse_memory_events(trace_data: Dict[str, Any]) -> List[Dict[str, Any]]:
    out: List[Dict[str, Any]] = []
    for evt in trace_data.get("traceEvents", []):
        name = str(evt.get("name", ""))
        if not name.startswith("Memory"):
            continue
        args = evt.get("args", {}) if isinstance(evt.get("args"), dict) else {}
        out.append(
            {
                "type": name,
                "timestamp_us": evt.get("ts", 0),
                "size": args.get("Bytes", args.get("size", 0)),
                "address": args.get("Addr", args.get("address", 0)),
                "process_id": evt.get("pid", 0),
                "thread_id": evt.get("tid", 0),
            }
        )
    return out


def parse_cuda_events(trace_data: Dict[str, Any]) -> List[Dict[str, Any]]:
    out: List[Dict[str, Any]] = []
    for evt in trace_data.get("traceEvents", []):
        cat = str(evt.get("cat", "")).lower()
        name = str(evt.get("name", ""))
        if cat != "cuda" and "cuda" not in name.lower():
            continue
        args = evt.get("args", {}) if isinstance(evt.get("args"), dict) else {}
        out.append(
            {
                "name": name,
                "start_time_us": evt.get("ts", 0),
                "duration_us": evt.get("dur", 0),
                "correlation_id": args.get("correlation", args.get("correlation_id", 0)),
                "device": args.get("device", 0),
                "stream": args.get("stream", 0),
            }
        )
    return out


def generate_summary(
    metadata: Dict[str, Any],
    operators: List[Dict[str, Any]],
    memory_events: List[Dict[str, Any]],
    cuda_events: List[Dict[str, Any]],
) -> Dict[str, Any]:
    categories: Dict[str, int] = defaultdict(int)
    for op in operators:
        categories[str(op.get("category", "unknown"))] += 1

    top_ops = sorted(operators, key=lambda x: x.get("duration_us", 0), reverse=True)[:20]

    current_memory = 0
    peak_memory = 0
    for evt in sorted(memory_events, key=lambda x: x.get("timestamp_us", 0)):
        evt_type = str(evt.get("type", "")).lower()
        size = int(evt.get("size", 0) or 0)
        if "alloc" in evt_type:
            current_memory += size
        elif "dealloc" in evt_type:
            current_memory -= size
        peak_memory = max(peak_memory, current_memory)

    kernel_stats: Dict[str, Dict[str, float]] = defaultdict(lambda: {"count": 0, "total_duration_us": 0.0})
    for evt in cuda_events:
        name = str(evt.get("name", ""))
        kernel_stats[name]["count"] += 1
        kernel_stats[name]["total_duration_us"] += float(evt.get("duration_us", 0))
    for name, stats in kernel_stats.items():
        stats["avg_duration_us"] = stats["total_duration_us"] / max(stats["count"], 1)

    return {
        "metadata": metadata,
        "total_operators": len(operators),
        "total_memory_events": len(memory_events),
        "total_cuda_events": len(cuda_events),
        "operator_categories": dict(categories),
        "top_operators": [
            {"name": op.get("name"), "category": op.get("category"), "duration_us": op.get("duration_us", 0)}
            for op in top_ops
        ],
        "memory_peak_bytes": peak_memory,
        "cuda_kernel_stats": dict(kernel_stats),
    }


def create_figures(summary: Dict[str, Any], operators: List[Dict[str, Any]], cuda_events: List[Dict[str, Any]], fig_dir: Path) -> None:
    fig_dir.mkdir(parents=True, exist_ok=True)
    fig, axes = plt.subplots(1, 2, figsize=(14, 5))

    if summary["operator_categories"]:
        cats = list(summary["operator_categories"].keys())
        vals = list(summary["operator_categories"].values())
        axes[0].pie(vals, labels=cats, autopct="%1.1f%%", startangle=90)
        axes[0].set_title("Operator Categories")
    else:
        axes[0].text(0.5, 0.5, "No operator events", ha="center", va="center")

    if operators:
        top = sorted(operators, key=lambda x: x.get("duration_us", 0), reverse=True)[:10]
        labels = [str(op.get("name", ""))[:28] for op in top]
        vals = [float(op.get("duration_us", 0)) for op in top]
        axes[1].bar(range(len(labels)), vals)
        axes[1].set_xticks(range(len(labels)))
        axes[1].set_xticklabels(labels, rotation=40, ha="right")
        axes[1].set_ylabel("Duration (us)")
        axes[1].set_title("Top 10 Operators by Duration")
    else:
        axes[1].text(0.5, 0.5, "No operator events", ha="center", va="center")

    plt.tight_layout()
    plt.savefig(fig_dir / "torch_trace_analysis.png", dpi=220, bbox_inches="tight")
    plt.close(fig)

    if cuda_events:
        plt.figure(figsize=(7, 5))
        durs = [float(e.get("duration_us", 0)) for e in cuda_events]
        plt.hist(durs, bins=60)
        plt.title("CUDA Event Duration Distribution")
        plt.xlabel("Duration (us)")
        plt.ylabel("Count")
        plt.yscale("log")
        plt.tight_layout()
        plt.savefig(fig_dir / "torch_cuda_duration_hist.png", dpi=220, bbox_inches="tight")
        plt.close()


def save_outputs(
    out_dir: Path,
    summary: Dict[str, Any],
    operators: List[Dict[str, Any]],
    memory_events: List[Dict[str, Any]],
    cuda_events: List[Dict[str, Any]],
    source_trace: Path,
) -> None:
    out_dir.mkdir(parents=True, exist_ok=True)

    if operators:
        pd.DataFrame(operators).to_csv(out_dir / "operators.csv", index=False)
    if memory_events:
        pd.DataFrame(memory_events).to_csv(out_dir / "memory_events.csv", index=False)
    if cuda_events:
        pd.DataFrame(cuda_events).to_csv(out_dir / "cuda_events.csv", index=False)

    payload = {
        "source_trace": str(source_trace),
        "summary": summary,
    }
    with open(out_dir / "trace_summary.json", "w", encoding="utf-8") as f:
        json.dump(payload, f, indent=2, ensure_ascii=False)

    with open(out_dir / "trace_summary.txt", "w", encoding="utf-8") as f:
        f.write("Torch Profiler Trace Summary\n")
        f.write("=" * 36 + "\n")
        f.write(f"Source trace: {source_trace}\n")
        f.write(f"Total operators: {summary['total_operators']}\n")
        f.write(f"Total memory events: {summary['total_memory_events']}\n")
        f.write(f"Total CUDA events: {summary['total_cuda_events']}\n")
        f.write(f"Peak memory: {summary['memory_peak_bytes']} bytes\n")


def main() -> None:
    parser = argparse.ArgumentParser(description="Parse torch profiler trace JSON")
    parser.add_argument("--trace-dir", default="results/traces/torch", help="Directory containing torch trace JSON")
    parser.add_argument("--trace-file", default=None, help="Optional explicit trace file")
    parser.add_argument("--output-dir", default="results/analysis/profiling/torch", help="Output table directory")
    parser.add_argument("--fig-dir", default="results/figures/profiling/torch", help="Output figure directory")
    args = parser.parse_args()

    trace_path = Path(args.trace_file) if args.trace_file else _pick_trace_file(Path(args.trace_dir))
    if trace_path is None or not trace_path.exists():
        raise FileNotFoundError(f"No torch trace json found (trace_dir={args.trace_dir}, trace_file={args.trace_file})")

    with open(trace_path, "r", encoding="utf-8") as f:
        trace_data = json.load(f)

    metadata = parse_metadata(trace_data)
    operators = parse_operators(trace_data)
    memory_events = parse_memory_events(trace_data)
    cuda_events = parse_cuda_events(trace_data)
    summary = generate_summary(metadata, operators, memory_events, cuda_events)

    create_figures(summary, operators, cuda_events, Path(args.fig_dir))
    save_outputs(Path(args.output_dir), summary, operators, memory_events, cuda_events, trace_path)
    print(f"[torch parse] done. outputs: {args.output_dir}")


if __name__ == "__main__":
    main()
