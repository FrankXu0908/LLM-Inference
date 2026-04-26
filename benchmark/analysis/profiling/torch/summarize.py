#!/usr/bin/env python3
"""Build a concise report from parsed torch profiling tables."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any, Dict

import matplotlib.pyplot as plt
import pandas as pd


def load_data(parsed_dir: Path) -> Dict[str, Any]:
    out: Dict[str, Any] = {}
    summary_file = parsed_dir / "trace_summary.json"
    if summary_file.exists():
        out["summary_payload"] = json.loads(summary_file.read_text(encoding="utf-8"))
        out["summary"] = out["summary_payload"].get("summary", {})
    if (parsed_dir / "operators.csv").exists():
        out["operators"] = pd.read_csv(parsed_dir / "operators.csv", low_memory=False)
    if (parsed_dir / "memory_events.csv").exists():
        out["memory"] = pd.read_csv(parsed_dir / "memory_events.csv")
    if (parsed_dir / "cuda_events.csv").exists():
        out["cuda"] = pd.read_csv(parsed_dir / "cuda_events.csv")
    return out


def build_report(data: Dict[str, Any]) -> Dict[str, Any]:
    summary = data.get("summary", {})
    report: Dict[str, Any] = {
        "totals": {
            "operators": int(summary.get("total_operators", 0)),
            "memory_events": int(summary.get("total_memory_events", 0)),
            "cuda_events": int(summary.get("total_cuda_events", 0)),
            "peak_memory_bytes": int(summary.get("memory_peak_bytes", 0)),
        },
        "top_operators": [],
        "top_cuda_events": [],
        "notes": [],
    }

    if "operators" in data and not data["operators"].empty:
        df = data["operators"].copy()
        if "duration_us" in df.columns:
            top_ops = df.nlargest(15, "duration_us")[["name", "category", "duration_us"]]
            report["top_operators"] = top_ops.to_dict(orient="records")
        op_count = int(df["name"].nunique()) if "name" in df.columns else 0
        report["notes"].append(f"Unique operator names: {op_count}")

    if "cuda" in data and not data["cuda"].empty:
        df = data["cuda"].copy()
        if "duration_us" in df.columns:
            top_cuda = df.nlargest(15, "duration_us")[["name", "duration_us"]]
            report["top_cuda_events"] = top_cuda.to_dict(orient="records")
            total_cuda = float(df["duration_us"].sum())
            avg_cuda = float(df["duration_us"].mean())
            report["notes"].append(f"CUDA duration total={total_cuda:.2f}us avg={avg_cuda:.2f}us")

    if "memory" in data and not data["memory"].empty:
        df = data["memory"]
        alloc = int(df["type"].astype(str).str.contains("alloc", case=False, na=False).sum()) if "type" in df.columns else 0
        dealloc = int(df["type"].astype(str).str.contains("dealloc", case=False, na=False).sum()) if "type" in df.columns else 0
        report["notes"].append(f"Memory alloc/dealloc events: {alloc}/{dealloc}")

    return report


def create_figure(report: Dict[str, Any], fig_path: Path) -> None:
    fig_path.parent.mkdir(parents=True, exist_ok=True)
    top = report.get("top_operators", [])[:10]
    if not top:
        return
    labels = [str(x["name"])[:28] for x in top]
    values = [float(x["duration_us"]) for x in top]
    plt.figure(figsize=(10, 5))
    plt.bar(range(len(labels)), values)
    plt.xticks(range(len(labels)), labels, rotation=35, ha="right")
    plt.ylabel("Duration (us)")
    plt.title("Top Torch Operators")
    plt.tight_layout()
    plt.savefig(fig_path, dpi=220, bbox_inches="tight")
    plt.close()


def main() -> None:
    parser = argparse.ArgumentParser(description="Summarize parsed torch profiling outputs")
    parser.add_argument("--parsed-dir", default="results/analysis/profiling/torch", help="Directory from parse_trace.py")
    parser.add_argument("--output-dir", default="results/analysis/profiling/torch", help="Summary output directory")
    parser.add_argument("--fig-path", default="results/figures/profiling/torch/torch_summary_top_ops.png")
    args = parser.parse_args()

    parsed_dir = Path(args.parsed_dir)
    if not parsed_dir.exists():
        raise FileNotFoundError(f"parsed dir not found: {parsed_dir}")

    data = load_data(parsed_dir)
    if not data:
        raise RuntimeError(f"no parsed files found under {parsed_dir}")

    report = build_report(data)
    out_dir = Path(args.output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    with open(out_dir / "profiling_report.json", "w", encoding="utf-8") as f:
        json.dump(report, f, indent=2, ensure_ascii=False)

    with open(out_dir / "profiling_report.txt", "w", encoding="utf-8") as f:
        f.write("Torch Profiling Report\n")
        f.write("=" * 24 + "\n")
        f.write(json.dumps(report, indent=2, ensure_ascii=False))

    create_figure(report, Path(args.fig_path))
    print(f"[torch summary] done. outputs: {out_dir}")


if __name__ == "__main__":
    main()
