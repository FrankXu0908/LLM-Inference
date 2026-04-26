#!/usr/bin/env python3
import argparse
import csv
import io
import json
import re
import subprocess
from pathlib import Path
from typing import Dict, List


COMM_PATTERNS = (
    r"\bnccl\b",
    r"allreduce",
    r"allgather",
    r"reduce_scatter",
    r"all_to_all",
    r"alltoall",
    r"broadcast",
)


def run_nsys_stats(rep: Path, report: str) -> str:
    report_candidates = [report]
    if report == "gpukernsum":
        report_candidates.append("cuda_gpu_kern_sum")
    elif report == "cudaapisum":
        report_candidates.append("cuda_api_sum")
    elif report == "nvtxsum":
        report_candidates.append("nvtx_sum")

    last_err = None
    for rep_name in report_candidates:
        cmd = [
            "nsys",
            "stats",
            "--force-export=true",
            "--report",
            rep_name,
            "--format",
            "csv",
            str(rep),
        ]
        proc = subprocess.run(cmd, capture_output=True, text=True)
        if proc.returncode == 0 and "Time (%),Total Time" in proc.stdout:
            return proc.stdout
        last_err = proc.stderr or proc.stdout
    raise RuntimeError(
        f"nsys stats failed for {rep} report={report_candidates}: {last_err}"
    )


def parse_csv_from_output(text: str) -> List[dict]:
    lines = text.splitlines()
    start = None
    for i, line in enumerate(lines):
        normalized = line.replace('"', "")
        if normalized.startswith("Time (%),Total Time"):
            start = i
            break
    if start is None:
        return []
    csv_text = "\n".join(lines[start:])
    return list(csv.DictReader(io.StringIO(csv_text)))


def as_float(v: str) -> float:
    try:
        return float((v or "0").replace(",", ""))
    except Exception:
        return 0.0


def is_comm_kernel(name: str) -> bool:
    n = (name or "").lower()
    return any(re.search(p, n) for p in COMM_PATTERNS)


def summarize_one_trace(rep: Path) -> Dict:
    kernel_rows = parse_csv_from_output(run_nsys_stats(rep, "gpukernsum"))

    gpu_total_ns = 0.0
    comm_total_ns = 0.0
    comm_kernels = []
    non_comm_kernels = []
    for r in kernel_rows:
        name = (r.get("Name") or "").strip()
        total_ns = as_float(r.get("Total Time (ns)", "0"))
        gpu_total_ns += total_ns
        item = {
            "name": name,
            "total_time_ns": total_ns,
            "time_pct_of_gpu": as_float(r.get("Time (%)", "0")),
            "calls": int(as_float(r.get("Instances", "0"))),
        }
        if is_comm_kernel(name):
            comm_total_ns += total_ns
            comm_kernels.append(item)
        else:
            non_comm_kernels.append(item)

    comm_kernels.sort(key=lambda x: x["total_time_ns"], reverse=True)
    non_comm_kernels.sort(key=lambda x: x["total_time_ns"], reverse=True)

    comm_pct = (comm_total_ns / gpu_total_ns * 100.0) if gpu_total_ns > 0 else 0.0
    compute_pct = 100.0 - comm_pct if gpu_total_ns > 0 else 0.0

    return {
        "trace": str(rep),
        "gpu_total_ns": gpu_total_ns,
        "comm_total_ns": comm_total_ns,
        "comm_pct_of_gpu": comm_pct,
        "compute_pct_of_gpu": compute_pct,
        "top_comm_kernels": comm_kernels[:10],
        "top_non_comm_kernels": non_comm_kernels[:10],
        "top_cuda_api_wait": [],
    }


def critical_path_conclusion(prefill: Dict, decode: Dict) -> Dict:
    pref = prefill["comm_pct_of_gpu"]
    dec = decode["comm_pct_of_gpu"]
    if dec - pref > 8:
        verdict = "decode is more TP-communication-bound than prefill"
    elif pref - dec > 8:
        verdict = "prefill is more TP-communication-bound than decode"
    else:
        verdict = "prefill/decode have similar TP communication pressure"
    return {
        "verdict": verdict,
        "prefill_comm_pct_of_gpu": pref,
        "decode_comm_pct_of_gpu": dec,
        "delta_decode_minus_prefill_pct": dec - pref,
    }


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Quantify TP communication and critical path from prefill/decode Nsight traces"
    )
    parser.add_argument("--prefill-trace", type=Path, required=True)
    parser.add_argument("--decode-trace", type=Path, default=None)
    parser.add_argument("--output-json", type=Path, required=True)
    args = parser.parse_args()

    prefill_summary = summarize_one_trace(args.prefill_trace)
    payload = {"prefill": prefill_summary}
    if args.decode_trace is not None:
        decode_summary = summarize_one_trace(args.decode_trace)
        critical = critical_path_conclusion(prefill_summary, decode_summary)
        payload["decode"] = decode_summary
        payload["critical_path"] = critical
    args.output_json.parent.mkdir(parents=True, exist_ok=True)
    with open(args.output_json, "w", encoding="utf-8") as f:
        json.dump(payload, f, indent=2, ensure_ascii=False)
    print(f"Saved TP communication summary to {args.output_json}")


if __name__ == "__main__":
    main()
