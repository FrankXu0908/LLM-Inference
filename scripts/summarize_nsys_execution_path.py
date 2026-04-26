#!/usr/bin/env python3
import argparse
import csv
import io
import json
import re
import subprocess
from pathlib import Path


COMM_PATTERNS = (
    r"\bnccl\b",
    r"allreduce",
    r"allgather",
    r"reduce_scatter",
    r"all_to_all",
)

KV_PATTERNS = (
    r"\bkv\b",
    r"cache",
    r"reshape_and_cache",
    r"zero_kv",
    r"slot_mapping",
)


def run_nsys_stats(rep: Path, report: str) -> str:
    cmd = [
        "nsys",
        "stats",
        "--force-export=true",
        "--report",
        report,
        "--format",
        "csv",
        str(rep),
    ]
    proc = subprocess.run(cmd, capture_output=True, text=True, check=True)
    return proc.stdout


def parse_csv_from_output(text: str, header_prefix: str) -> list[dict]:
    lines = text.splitlines()
    start = None
    for i, line in enumerate(lines):
        if line.startswith(header_prefix):
            start = i
            break
    if start is None:
        return []
    csv_text = "\n".join(lines[start:])
    reader = csv.DictReader(io.StringIO(csv_text))
    return list(reader)


def as_float(row: dict, key: str) -> float:
    try:
        return float((row.get(key) or "0").replace(",", ""))
    except Exception:
        return 0.0


def classify_kernel(name: str) -> str:
    n = name.lower()
    if any(re.search(p, n) for p in COMM_PATTERNS):
        return "communication"
    if any(re.search(p, n) for p in KV_PATTERNS):
        return "kv_memory"
    return "other_gpu"


def main() -> None:
    parser = argparse.ArgumentParser(description="Summarize execution path from nsys trace")
    parser.add_argument("--trace", type=Path, required=True, help="Path to .nsys-rep")
    parser.add_argument("--output-json", type=Path, required=True)
    args = parser.parse_args()

    # 1) NVTX ranges for prefill/decode wall-time share.
    nvtx_text = run_nsys_stats(args.trace, "nvtxsum")
    nvtx_rows = parse_csv_from_output(nvtx_text, "Time (%),Total Time")
    prefill_ns = 0.0
    decode_ns = 0.0
    for r in nvtx_rows:
        name = (r.get("Range") or r.get("Name") or "").strip()
        total_ns = as_float(r, "Total Time (ns)")
        if name == "PHASE_PREFILL":
            prefill_ns += total_ns
        elif name == "PHASE_DECODE":
            decode_ns += total_ns
    phase_total_ns = max(prefill_ns + decode_ns, 1e-9)

    # 2) GPU kernels for communication/KV split.
    kernel_text = run_nsys_stats(args.trace, "gpukernsum")
    kernel_rows = parse_csv_from_output(kernel_text, "Time (%),Total Time")
    comm_ns = 0.0
    kv_ns = 0.0
    other_gpu_ns = 0.0
    for r in kernel_rows:
        name = (r.get("Name") or "").strip()
        total_ns = as_float(r, "Total Time (ns)")
        cat = classify_kernel(name)
        if cat == "communication":
            comm_ns += total_ns
        elif cat == "kv_memory":
            kv_ns += total_ns
        else:
            other_gpu_ns += total_ns
    gpu_total_ns = max(comm_ns + kv_ns + other_gpu_ns, 1e-9)

    payload = {
        "trace": str(args.trace),
        "breakdown_pct": {
            "prefill": prefill_ns / phase_total_ns * 100.0,
            "decode": decode_ns / phase_total_ns * 100.0,
            "kv_memory": kv_ns / gpu_total_ns * 100.0,
            "communication": comm_ns / gpu_total_ns * 100.0,
            "other_gpu": other_gpu_ns / gpu_total_ns * 100.0,
        },
        "raw_ns": {
            "phase_prefill_ns": prefill_ns,
            "phase_decode_ns": decode_ns,
            "gpu_kv_ns": kv_ns,
            "gpu_comm_ns": comm_ns,
            "gpu_other_ns": other_gpu_ns,
        },
    }

    args.output_json.parent.mkdir(parents=True, exist_ok=True)
    with open(args.output_json, "w", encoding="utf-8") as f:
        json.dump(payload, f, indent=2, ensure_ascii=False)
    print(f"Saved summary to {args.output_json}")


if __name__ == "__main__":
    main()
