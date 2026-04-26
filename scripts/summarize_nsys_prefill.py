#!/usr/bin/env python3
import argparse
import csv
import io
import json
import re
import subprocess
from pathlib import Path
from typing import Dict, List


ATTN_PATTERNS = (
    r"flash",
    r"fused_attention",
    r"attention",
    r"attn",
    r"fmha",
    r"sdpa",
    r"scaled[_ ]dot[_ ]product",
)
GEMM_PATTERNS = (
    r"gemm",
    r"cublas",
    r"cutlass",
    r"mma",
    r"matmul",
    r"sgemm",
    r"hgemm",
    r"igemm",
)


def classify_kernel(name: str) -> str:
    n = name.lower()
    if any(re.search(p, n) for p in ATTN_PATTERNS):
        return "attention"
    if any(re.search(p, n) for p in GEMM_PATTERNS):
        return "mlp_gemm"
    return "other"


def parse_gpukernsum_csv(text: str) -> List[Dict]:
    lines = text.splitlines()
    start = None
    for i, line in enumerate(lines):
        if line.startswith("Time (%),Total Time"):
            start = i
            break
    if start is None:
        raise RuntimeError("Cannot find gpukernsum CSV header in nsys output")

    csv_text = "\n".join(lines[start:])
    reader = csv.DictReader(io.StringIO(csv_text))
    rows = []
    for row in reader:
        try:
            total_ns = float(row.get("Total Time (ns)", "0") or 0)
        except ValueError:
            total_ns = 0.0
        name = (row.get("Name") or "").strip()
        if not name or total_ns <= 0:
            continue
        rows.append({"name": name, "total_ns": total_ns})
    return rows


def summarize_trace(rep_path: Path) -> Dict:
    cmd = [
        "nsys",
        "stats",
        "--force-export=true",
        "--report",
        "gpukernsum",
        "--format",
        "csv",
        str(rep_path),
    ]
    proc = subprocess.run(cmd, capture_output=True, text=True, check=True)
    rows = parse_gpukernsum_csv(proc.stdout)

    buckets_ns = {"attention": 0.0, "mlp_gemm": 0.0, "other": 0.0}
    for row in rows:
        buckets_ns[classify_kernel(row["name"])] += row["total_ns"]

    total_ns = sum(buckets_ns.values())
    ratio = {
        k: (v / total_ns * 100.0 if total_ns > 0 else 0.0)
        for k, v in buckets_ns.items()
    }
    return {"cuda_time_ns": buckets_ns, "ratio_pct": ratio}


def main() -> None:
    parser = argparse.ArgumentParser(description="Summarize nsys traces for prefill attention/MLP split")
    parser.add_argument("--trace-dir", type=Path, required=True)
    parser.add_argument("--lengths", type=int, nargs="+", default=[256, 2048, 4096, 8192])
    parser.add_argument("--output-json", type=Path, required=True)
    args = parser.parse_args()

    results = []
    for length in args.lengths:
        rep = args.trace_dir / f"prefill_{length}.nsys-rep"
        if not rep.exists():
            raise FileNotFoundError(f"Missing trace: {rep}")
        summary = summarize_trace(rep)
        results.append({"input_tokens": length, **summary})

    args.output_json.parent.mkdir(parents=True, exist_ok=True)
    payload = {
        "trace_dir": str(args.trace_dir),
        "results": results,
    }
    with open(args.output_json, "w", encoding="utf-8") as f:
        json.dump(payload, f, indent=2, ensure_ascii=False)
    print(f"Saved summary to {args.output_json}")


if __name__ == "__main__":
    main()
