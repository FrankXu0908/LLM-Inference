#!/usr/bin/env python3
"""Standalone same-head FA1/FA2 forward microbenchmark.

This benchmark intentionally uses q_heads == kv_heads so FlashAttention-1 and
FlashAttention-2 can be compared without GQA support differences.
"""

from __future__ import annotations

import argparse
import json
import math
import statistics
import sys
from pathlib import Path
from typing import Any, Callable

import torch


def percentile(values: list[float], pct: float) -> float:
    ordered = sorted(values)
    idx = (len(ordered) - 1) * pct / 100.0
    low = math.floor(idx)
    high = math.ceil(idx)
    if low == high:
        return ordered[int(idx)]
    return ordered[low] * (high - idx) + ordered[high] * (idx - low)


def causal_attention_flops(batch: int, seq_len: int, heads: int, head_dim: int) -> int:
    return 2 * batch * heads * seq_len * (seq_len + 1) * head_dim


def tensor_bytes(batch: int, seq_len: int, heads: int, head_dim: int, dtype: torch.dtype) -> int:
    elem_size = torch.empty((), dtype=dtype).element_size()
    one = batch * seq_len * heads * head_dim * elem_size
    return 4 * one  # q, k, v, out


def load_backend(backend: str, flash_attn_source: str | None, fa2_interface: str) -> Callable:
    if flash_attn_source:
        sys.path.insert(0, flash_attn_source)

    if backend == "fa2":
        if fa2_interface == "dense":
            try:
                from flash_attn import flash_attn_func
            except Exception as exc:
                raise RuntimeError("Could not import FA2 flash_attn_func.") from exc

            def run(q: torch.Tensor, k: torch.Tensor, v: torch.Tensor, batch: int, seq_len: int):
                q4 = q.view(batch, seq_len, q.shape[1], q.shape[2])
                k4 = k.view(batch, seq_len, k.shape[1], k.shape[2])
                v4 = v.view(batch, seq_len, v.shape[1], v.shape[2])
                return flash_attn_func(q4, k4, v4, dropout_p=0.0, causal=True)

            return run

        try:
            from flash_attn import flash_attn_varlen_func
        except Exception as exc:
            raise RuntimeError("Could not import FA2 flash_attn_varlen_func.") from exc

        def run(q: torch.Tensor, k: torch.Tensor, v: torch.Tensor, batch: int, seq_len: int):
            cu = torch.arange(0, (batch + 1) * seq_len, seq_len, device=q.device, dtype=torch.int32)
            return flash_attn_varlen_func(
                q,
                k,
                v,
                cu,
                cu,
                seq_len,
                seq_len,
                dropout_p=0.0,
                causal=True,
            )

        return run

    if backend == "fa1":
        try:
            from flash_attn.flash_attn_interface import flash_attn_unpadded_func
        except Exception as exc:
            raise RuntimeError("Could not import FA1 flash_attn_unpadded_func.") from exc

        def run(q: torch.Tensor, k: torch.Tensor, v: torch.Tensor, batch: int, seq_len: int):
            cu = torch.arange(0, (batch + 1) * seq_len, seq_len, device=q.device, dtype=torch.int32)
            return flash_attn_unpadded_func(
                q,
                k,
                v,
                cu,
                cu,
                seq_len,
                seq_len,
                0.0,
                causal=True,
            )

        return run

    raise ValueError(f"Unsupported backend: {backend}")


def run_one(
    fn: Callable,
    *,
    backend: str,
    batch: int,
    seq_len: int,
    heads: int,
    head_dim: int,
    dtype: torch.dtype,
    warmup: int,
    iters: int,
    seed: int,
) -> dict[str, Any]:
    torch.manual_seed(seed)
    torch.cuda.empty_cache()
    torch.cuda.reset_peak_memory_stats()

    total = batch * seq_len
    q = torch.randn((total, heads, head_dim), device="cuda", dtype=dtype)
    k = torch.randn((total, heads, head_dim), device="cuda", dtype=dtype)
    v = torch.randn((total, heads, head_dim), device="cuda", dtype=dtype)

    with torch.inference_mode():
        for _ in range(warmup):
            out = fn(q, k, v, batch, seq_len)
        torch.cuda.synchronize()

        times_ms: list[float] = []
        for _ in range(iters):
            start = torch.cuda.Event(enable_timing=True)
            end = torch.cuda.Event(enable_timing=True)
            start.record()
            out = fn(q, k, v, batch, seq_len)
            end.record()
            torch.cuda.synchronize()
            times_ms.append(float(start.elapsed_time(end)))

    checksum = float(out.float().mean().detach().cpu())
    median_ms = statistics.median(times_ms)
    flops = causal_attention_flops(batch, seq_len, heads, head_dim)
    bytes_moved = tensor_bytes(batch, seq_len, heads, head_dim, dtype)
    return {
        "backend": backend,
        "batch": batch,
        "seq_len": seq_len,
        "heads": heads,
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
    parser = argparse.ArgumentParser(description="Same-head FA1/FA2 CUDA benchmark")
    parser.add_argument("--backend", choices=["fa1", "fa2"], required=True)
    parser.add_argument("--fa2-interface", choices=["varlen", "dense"], default="varlen")
    parser.add_argument("--flash-attn-source", default=None)
    parser.add_argument("--output-json", type=Path, required=True)
    parser.add_argument("--batches", type=int, nargs="+", default=[1, 4, 16])
    parser.add_argument("--seq-lens", type=int, nargs="+", default=[512, 2048, 8192])
    parser.add_argument("--heads", type=int, default=32)
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

    dtype = torch.bfloat16 if args.dtype == "bf16" else torch.float16
    fn = load_backend(args.backend, args.flash_attn_source, args.fa2_interface)
    capability = torch.cuda.get_device_capability()
    results: list[dict[str, Any]] = []
    for batch in args.batches:
        for seq_len in args.seq_lens:
            print(f"{args.backend.upper()} same-head benchmark: batch={batch} seq={seq_len}")
            result = run_one(
                fn,
                backend=args.backend,
                batch=batch,
                seq_len=seq_len,
                heads=args.heads,
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
        "experiment": "same_head_flash_attention_forward",
        "backend": args.backend,
        "fa2_interface": args.fa2_interface if args.backend == "fa2" else None,
        "gpu": torch.cuda.get_device_name(),
        "compute_capability": f"sm{capability[0]}{capability[1]}",
        "results": results,
    }
    args.output_json.parent.mkdir(parents=True, exist_ok=True)
    args.output_json.write_text(json.dumps(payload, indent=2), encoding="utf-8")
    print(f"Saved {args.output_json}")


if __name__ == "__main__":
    main()
