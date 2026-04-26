#!/usr/bin/env python3
"""One-command profiling post-analysis pipeline (torch + nsys).

Important:
- This script does NOT collect traces.
- It requires pre-collected profiling traces (torch JSON and/or NSYS export).
"""

from __future__ import annotations

import argparse
import subprocess
import sys
from pathlib import Path


def _run(cmd: list[str]) -> None:
    print("[run]", " ".join(cmd))
    subprocess.run(cmd, check=True)


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Run profiling post-analysis pipeline (requires existing traces)"
    )
    parser.add_argument("--torch-trace-dir", default="results/traces/torch")
    parser.add_argument("--torch-trace-file", default=None)
    parser.add_argument("--nsys-trace-dir", default="results/traces/nsys")
    parser.add_argument("--nsys-trace-file", default=None)
    parser.add_argument("--output-root", default="results/analysis/profiling")
    parser.add_argument("--figure-root", default="results/figures/profiling")
    parser.add_argument(
        "--skip-missing",
        action="store_true",
        help="Skip branch if trace missing instead of failing",
    )
    args = parser.parse_args()

    root = Path(__file__).resolve().parent
    py = sys.executable

    torch_out = Path(args.output_root) / "torch"
    torch_fig = Path(args.figure_root) / "torch"
    nsys_out = Path(args.output_root) / "nsys"
    nsys_fig = Path(args.figure_root) / "nsys"

    print(
        "[info] run_pipeline.py analyzes existing traces only. "
        "If traces are missing, collect them first, then rerun this command."
    )

    # Torch branch
    torch_parse = [
        py,
        str(root / "torch" / "parse_trace.py"),
        "--trace-dir",
        args.torch_trace_dir,
        "--output-dir",
        str(torch_out),
        "--fig-dir",
        str(torch_fig),
    ]
    if args.torch_trace_file:
        torch_parse.extend(["--trace-file", args.torch_trace_file])

    try:
        _run(torch_parse)
        _run(
            [
                py,
                str(root / "torch" / "summarize.py"),
                "--parsed-dir",
                str(torch_out),
                "--output-dir",
                str(torch_out),
                "--fig-path",
                str(torch_fig / "torch_summary_top_ops.png"),
            ]
        )
    except Exception:
        if args.skip_missing:
            print("[warn] torch pipeline skipped")
        else:
            raise

    # NSYS branch
    nsys_parse = [
        py,
        str(root / "nsys" / "parse_nsys.py"),
        "--trace-dir",
        args.nsys_trace_dir,
        "--output-dir",
        str(nsys_out),
        "--fig-dir",
        str(nsys_fig),
    ]
    if args.nsys_trace_file:
        nsys_parse.extend(["--trace-file", args.nsys_trace_file])

    try:
        _run(nsys_parse)
        _run(
            [
                py,
                str(root / "nsys" / "kernel_stats.py"),
                "--data-dir",
                str(nsys_out),
                "--output-dir",
                str(nsys_out),
                "--fig-path",
                str(nsys_fig / "kernel_analysis.png"),
            ]
        )
    except Exception:
        if args.skip_missing:
            print("[warn] nsys pipeline skipped")
        else:
            raise

    print("[done] profiling analysis pipeline completed")


if __name__ == "__main__":
    main()
