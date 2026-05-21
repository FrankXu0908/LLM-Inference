#!/usr/bin/env python3
"""Standalone FlashAttention-2 CUDA forward microbenchmark.

This script intentionally bypasses vLLM. It calls flash-attn's CUDA FA2
interface directly so kernel measurements are not mixed with scheduler, paged
KV cache, HTTP serving, or request batching effects.
"""

from __future__ import annotations

import argparse
import json
import math
import statistics
import sys
from pathlib import Path
from typing import Any

import torch


def import_flash_attn(source_dir: str | None):
    if source_dir:
        sys.path.insert(0, source_dir)
    try:
        from flash_attn import flash_attn_func
    except Exception as exc:  # pragma: no cover - this is an environment check.
        raise RuntimeError(
            "Could not import flash_attn.flash_attn_func. Build/install the CUDA "
            "flash-attn package first, for example:\n"
            "  cd /home/xuliren/repo/flash-attention\n"
            "  pip install -e . --no-build-isolation\n"
            "Then rerun this script in the same conda environment."
        ) from exc
    return flash_attn_func


def percentile(values: list[float], pct: float) -> float:
    if not values:
        return 0.0
    ordered = sorted(values)
    idx = (len(ordered) - 1) * pct / 100.0
    low = math.floor(idx)
    high = math.ceil(idx)
    if low == high:
        return ordered[int(idx)]
    return ordered[low] * (high - idx) + ordered[high] * (idx - low)


def causal_attention_flops(batch: int, seq_len: int, q_heads: int, head_dim: int) -> int:
    # QK^T and P@V each cost roughly 2 flops per multiply-add. For causal,
    # each query attends to an average of (seq_len + 1) / 2 keys.
    return 2 * batch * q_heads * seq_len * (seq_len + 1) * head_dim


def tensor_bytes(batch: int, seq_len: int, q_heads: int, kv_heads: int, head_dim: int, dtype: torch.dtype) -> int:
    elem_size = torch.empty((), dtype=dtype).element_size()
    q_bytes = batch * seq_len * q_heads * head_dim * elem_size
    k_bytes = batch * seq_len * kv_heads * head_dim * elem_size
    v_bytes = k_bytes
    out_bytes = q_bytes
    return q_bytes + k_bytes + v_bytes + out_bytes


def run_one(
    flash_attn_func,
    *,
    batch: int,
    seq_len: int,
    q_heads: int,
    kv_heads: int,
    head_dim: int,
    dtype: torch.dtype,
    warmup: int,
    iters: int,
    seed: int,
) -> dict[str, Any]:
    torch.manual_seed(seed)
    torch.cuda.empty_cache()
    torch.cuda.reset_peak_memory_stats()

    q = torch.randn((batch, seq_len, q_heads, head_dim), device="cuda", dtype=dtype)
    k = torch.randn((batch, seq_len, kv_heads, head_dim), device="cuda", dtype=dtype)
    v = torch.randn((batch, seq_len, kv_heads, head_dim), device="cuda", dtype=dtype)

    with torch.inference_mode():
        for _ in range(warmup):
            out = flash_attn_func(q, k, v, dropout_p=0.0, causal=True)
        torch.cuda.synchronize()

        times_ms: list[float] = []
        for _ in range(iters):
            start = torch.cuda.Event(enable_timing=True)
            end = torch.cuda.Event(enable_timing=True)
            start.record()
            out = flash_attn_func(q, k, v, dropout_p=0.0, causal=True)
            end.record()
            torch.cuda.synchronize()
            times_ms.append(float(start.elapsed_time(end)))

    # Keep output live until timing is complete.
    checksum = float(out.float().mean().detach().cpu())

    median_ms = statistics.median(times_ms)
    flops = causal_attention_flops(batch, seq_len, q_heads, head_dim)
    bytes_moved = tensor_bytes(batch, seq_len, q_heads, kv_heads, head_dim, dtype)
    return {
        "batch": batch,
        "seq_len": seq_len,
        "q_heads": q_heads,
        "kv_heads": kv_heads,
        "head_dim": head_dim,
        "dtype": str(dtype).replace("torch.", ""),
        "causal": True,
        "dropout_p": 0.0,
        "warmup": warmup,
        "iters": iters,
        "latency_ms": {
            "median": median_ms,
            "mean": statistics.fmean(times_ms),
            "p90": percentile(times_ms, 90),
            "p99": percentile(times_ms, 99),
            "min": min(times_ms),
            "max": max(times_ms),
        },
        "tflops_est": flops / (median_ms / 1000.0) / 1e12,
        "io_gb_est": bytes_moved / 1e9,
        "effective_bandwidth_gbs_est": bytes_moved / (median_ms / 1000.0) / 1e9,
        "max_memory_allocated_gb": torch.cuda.max_memory_allocated() / 1e9,
        "checksum": checksum,
    }


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Standalone FlashAttention-2 CUDA benchmark")
    parser.add_argument("--flash-attn-source", default="/home/xuliren/repo/flash-attention")
    parser.add_argument("--output-json", type=Path, required=True)
    parser.add_argument("--batches", type=int, nargs="+", default=[1, 4, 16])
    parser.add_argument("--seq-lens", type=int, nargs="+", default=[512, 2048, 8192, 16384])
    parser.add_argument("--q-heads", type=int, default=32)
    parser.add_argument("--kv-heads", type=int, default=8)
    parser.add_argument("--head-dim", type=int, default=128)
    parser.add_argument("--dtype", choices=["bf16", "fp16"], default="bf16")
    parser.add_argument("--warmup", type=int, default=5)
    parser.add_argument("--iters", type=int, default=20)
    parser.add_argument("--seed", type=int, default=42)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    if not torch.cuda.is_available():
        raise RuntimeError("CUDA is not available in this environment.")

    flash_attn_func = import_flash_attn(args.flash_attn_source)
    dtype = torch.bfloat16 if args.dtype == "bf16" else torch.float16
    device_name = torch.cuda.get_device_name()
    capability = torch.cuda.get_device_capability()

    results: list[dict[str, Any]] = []
    for batch in args.batches:
        for seq_len in args.seq_lens:
            print(f"FA2 CUDA benchmark: batch={batch} seq={seq_len}")
            result = run_one(
                flash_attn_func,
                batch=batch,
                seq_len=seq_len,
                q_heads=args.q_heads,
                kv_heads=args.kv_heads,
                head_dim=args.head_dim,
                dtype=dtype,
                warmup=args.warmup,
                iters=args.iters,
                seed=args.seed,
            )
            results.append(result)
            lat = result["latency_ms"]
            print(
                f"  median={lat['median']:.3f} ms "
                f"p90={lat['p90']:.3f} ms "
                f"TFLOP/s={result['tflops_est']:.2f} "
                f"max_mem={result['max_memory_allocated_gb']:.2f} GB"
            )

    payload = {
        "experiment": "standalone_flash_attention_2_cuda_forward",
        "source": args.flash_attn_source,
        "gpu": device_name,
        "compute_capability": f"sm{capability[0]}{capability[1]}",
        "fixed_traits_from_source_for_qwen3_8b_sm89_causal_hdim128": {
            "head_dim": 128,
            "block_m": 64,
            "block_n": 64,
            "warps": 4,
            "dropout": False,
            "causal": True,
            "dtype": "bf16",
        },
        "results": results,
    }
    args.output_json.parent.mkdir(parents=True, exist_ok=True)
    with open(args.output_json, "w", encoding="utf-8") as f:
        json.dump(payload, f, indent=2)
    print(f"Saved {args.output_json}")


if __name__ == "__main__":
    main()
