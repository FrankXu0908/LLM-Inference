#!/usr/bin/env python3
"""Analyze parsed NSYS kernel/memory/api tables and emit actionable stats."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any, Dict, List

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd


def load_data(data_dir: Path) -> Dict[str, Any]:
    out: Dict[str, Any] = {}
    summary_file = data_dir / "nsys_summary.json"
    if summary_file.exists():
        payload = json.loads(summary_file.read_text(encoding="utf-8"))
        out["summary_payload"] = payload
        out["summary"] = payload.get("summary", {})
    if (data_dir / "gpu_kernels.csv").exists():
        out["kernels"] = pd.read_csv(data_dir / "gpu_kernels.csv")
    if (data_dir / "memory_operations.csv").exists():
        out["memory"] = pd.read_csv(data_dir / "memory_operations.csv")
    if (data_dir / "cuda_api_calls.csv").exists():
        out["api"] = pd.read_csv(data_dir / "cuda_api_calls.csv")
    return out


def analyze_kernel_occupancy(kernels: pd.DataFrame | None) -> Dict[str, Any]:
    result = {"occupancy_stats": {}, "bottlenecks": [], "optimization_opportunities": []}
    if kernels is None or kernels.empty:
        return result

    df = kernels.copy()
    if "registers" in df.columns:
        reg = pd.to_numeric(df["registers"], errors="coerce").dropna()
        if not reg.empty:
            result["occupancy_stats"]["avg_registers_per_thread"] = float(reg.mean())
            result["occupancy_stats"]["max_registers_per_thread"] = float(reg.max())
            if (reg > 255).any():
                result["bottlenecks"].append("High register usage (>255) observed")
                result["optimization_opportunities"].append("Reduce register pressure in hot kernels")

    if "shared_memory" in df.columns:
        shm = pd.to_numeric(df["shared_memory"], errors="coerce").dropna()
        if not shm.empty:
            result["occupancy_stats"]["avg_shared_memory_bytes"] = float(shm.mean())
            result["occupancy_stats"]["max_shared_memory_bytes"] = float(shm.max())

    if "duration_ns" in df.columns:
        durs = pd.to_numeric(df["duration_ns"], errors="coerce").dropna()
        if not durs.empty:
            result["occupancy_stats"]["avg_kernel_duration_us"] = float(durs.mean() / 1e3)

    return result


def analyze_memory(memory: pd.DataFrame | None) -> Dict[str, Any]:
    result = {"bandwidth_stats": {}, "access_patterns": {}, "bottlenecks": [], "recommendations": []}
    if memory is None or memory.empty:
        return result
    df = memory.copy()

    if "throughput" in df.columns:
        bw = pd.to_numeric(df["throughput"], errors="coerce").dropna()
        if not bw.empty:
            result["bandwidth_stats"] = {
                "avg_throughput_gbps": float(bw.mean()),
                "max_throughput_gbps": float(bw.max()),
                "min_throughput_gbps": float(bw.min()),
            }

    if "size" in df.columns:
        size = pd.to_numeric(df["size"], errors="coerce").dropna()
        if not size.empty:
            result["access_patterns"]["avg_transfer_size_bytes"] = float(size.mean())
            result["access_patterns"]["transfer_size_distribution"] = {
                "small_lt_1kb": int((size < 1024).sum()),
                "medium_1kb_1mb": int(((size >= 1024) & (size < 1024 * 1024)).sum()),
                "large_gt_1mb": int((size >= 1024 * 1024).sum()),
            }
            if (size < 4096).mean() > 0.5:
                result["recommendations"].append("Many tiny memory transfers; consider batching")

    if "source" in df.columns and "destination" in df.columns:
        pat = df.groupby(["source", "destination"]).size().sort_values(ascending=False).head(10)
        result["access_patterns"]["top_transfer_paths"] = {f"{k[0]}->{k[1]}": int(v) for k, v in pat.items()}
    return result


def analyze_launch(kernels: pd.DataFrame | None) -> Dict[str, Any]:
    result = {"launch_stats": {}, "optimization_suggestions": []}
    if kernels is None or kernels.empty:
        return result
    df = kernels.copy()
    result["launch_stats"]["total_launches"] = int(len(df))

    if "start_time_ns" in df.columns:
        starts = pd.to_numeric(df["start_time_ns"], errors="coerce").dropna().sort_values()
        intervals = starts.diff().dropna()
        if not intervals.empty:
            avg_interval = float(intervals.mean())
            result["launch_stats"]["avg_launch_interval_ns"] = avg_interval
            result["launch_stats"]["launch_frequency_hz"] = float(1e9 / avg_interval) if avg_interval > 0 else 0.0

    if "duration_ns" in df.columns:
        durs = pd.to_numeric(df["duration_ns"], errors="coerce").dropna()
        if not durs.empty and (durs < 1000).mean() > 0.3:
            result["optimization_suggestions"].append("Many sub-1us kernels; fusion likely useful")
    return result


def build_recommendations(occ: Dict[str, Any], mem: Dict[str, Any], launch: Dict[str, Any]) -> List[str]:
    recs = []
    recs.extend(occ.get("optimization_opportunities", []))
    recs.extend(mem.get("recommendations", []))
    recs.extend(launch.get("optimization_suggestions", []))
    recs.extend(
        [
            "Check overlap between compute kernels and NCCL to reduce exposed comm time",
            "Tune launch config for hottest kernels first",
            "Prioritize optimizations on top-duration kernels before tail cleanup",
        ]
    )
    seen = set()
    uniq = []
    for r in recs:
        if r not in seen:
            uniq.append(r)
            seen.add(r)
    return uniq


def create_figure(occ: Dict[str, Any], mem: Dict[str, Any], launch: Dict[str, Any], fig_path: Path) -> None:
    fig_path.parent.mkdir(parents=True, exist_ok=True)
    fig, axes = plt.subplots(1, 3, figsize=(16, 4))

    occ_stats = occ.get("occupancy_stats", {})
    if occ_stats:
        labels = list(occ_stats.keys())
        vals = [occ_stats[k] for k in labels]
        axes[0].bar(range(len(labels)), vals)
        axes[0].set_xticks(range(len(labels)))
        axes[0].set_xticklabels([x.replace("_", "\n") for x in labels], rotation=0)
        axes[0].set_title("Kernel Occupancy Stats")
    else:
        axes[0].text(0.5, 0.5, "No occupancy data", ha="center", va="center")

    mem_stats = mem.get("bandwidth_stats", {})
    if mem_stats:
        labels = list(mem_stats.keys())
        vals = [mem_stats[k] for k in labels]
        axes[1].bar(range(len(labels)), vals)
        axes[1].set_xticks(range(len(labels)))
        axes[1].set_xticklabels([x.replace("_", "\n") for x in labels], rotation=0)
        axes[1].set_title("Memory Throughput")
    else:
        axes[1].text(0.5, 0.5, "No memory stats", ha="center", va="center")

    launch_stats = launch.get("launch_stats", {})
    if launch_stats:
        labels = list(launch_stats.keys())
        vals = [launch_stats[k] for k in labels]
        axes[2].bar(range(len(labels)), vals)
        axes[2].set_xticks(range(len(labels)))
        axes[2].set_xticklabels([x.replace("_", "\n") for x in labels], rotation=0)
        axes[2].set_title("Launch Stats")
        axes[2].set_yscale("log")
    else:
        axes[2].text(0.5, 0.5, "No launch stats", ha="center", va="center")

    plt.tight_layout()
    plt.savefig(fig_path, dpi=220, bbox_inches="tight")
    plt.close(fig)


def save_report(out_dir: Path, occ: Dict[str, Any], mem: Dict[str, Any], launch: Dict[str, Any], recs: List[str]) -> None:
    out_dir.mkdir(parents=True, exist_ok=True)
    payload = {
        "occupancy": occ,
        "memory": mem,
        "launch": launch,
        "recommendations": recs,
    }
    (out_dir / "kernel_analysis_report.json").write_text(json.dumps(payload, indent=2, ensure_ascii=False), encoding="utf-8")
    with open(out_dir / "kernel_analysis_report.txt", "w", encoding="utf-8") as f:
        f.write("NSYS Kernel Analysis Report\n")
        f.write("=" * 28 + "\n")
        f.write(json.dumps(payload, indent=2, ensure_ascii=False))


def main() -> None:
    parser = argparse.ArgumentParser(description="Analyze parsed NSYS tables")
    parser.add_argument("--data-dir", default="results/analysis/profiling/nsys", help="Directory from parse_nsys.py")
    parser.add_argument("--output-dir", default="results/analysis/profiling/nsys", help="Output report directory")
    parser.add_argument("--fig-path", default="results/figures/profiling/nsys/kernel_analysis.png")
    args = parser.parse_args()

    data = load_data(Path(args.data_dir))
    if not data:
        raise RuntimeError(f"no parsed NSYS files found under {args.data_dir}")

    occ = analyze_kernel_occupancy(data.get("kernels"))
    mem = analyze_memory(data.get("memory"))
    launch = analyze_launch(data.get("kernels"))
    recs = build_recommendations(occ, mem, launch)

    save_report(Path(args.output_dir), occ, mem, launch, recs)
    create_figure(occ, mem, launch, Path(args.fig_path))
    print(f"[nsys stats] done. outputs: {args.output_dir}")


if __name__ == "__main__":
    main()

