#!/usr/bin/env python3
"""Run one FlashAttention-2 CUDA forward call for Nsight Compute profiling."""

from __future__ import annotations

import argparse
import sys

import torch


def import_flash_attn(source_dir: str | None):
    if source_dir:
        sys.path.insert(0, source_dir)
    try:
        from flash_attn import flash_attn_func
    except Exception as exc:
        raise RuntimeError(
            "Could not import flash_attn.flash_attn_func. Build/install "
            "/home/xuliren/repo/flash-attention first."
        ) from exc
    return flash_attn_func


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="One-shot FA2 CUDA forward for NCU")
    parser.add_argument("--flash-attn-source", default="/home/xuliren/repo/flash-attention")
    parser.add_argument("--batch", type=int, default=1)
    parser.add_argument("--seq-len", type=int, default=8192)
    parser.add_argument("--q-heads", type=int, default=32)
    parser.add_argument("--kv-heads", type=int, default=8)
    parser.add_argument("--head-dim", type=int, default=128)
    parser.add_argument("--dtype", choices=["bf16", "fp16"], default="bf16")
    parser.add_argument("--warmup", type=int, default=3)
    parser.add_argument("--seed", type=int, default=42)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    if not torch.cuda.is_available():
        raise RuntimeError("CUDA is not available in this environment.")

    flash_attn_func = import_flash_attn(args.flash_attn_source)
    dtype = torch.bfloat16 if args.dtype == "bf16" else torch.float16
    torch.manual_seed(args.seed)

    q = torch.randn((args.batch, args.seq_len, args.q_heads, args.head_dim), device="cuda", dtype=dtype)
    k = torch.randn((args.batch, args.seq_len, args.kv_heads, args.head_dim), device="cuda", dtype=dtype)
    v = torch.randn((args.batch, args.seq_len, args.kv_heads, args.head_dim), device="cuda", dtype=dtype)

    print(
        "FA2 one-shot: "
        f"batch={args.batch} seq={args.seq_len} q_heads={args.q_heads} "
        f"kv_heads={args.kv_heads} head_dim={args.head_dim} dtype={args.dtype}"
    )

    with torch.inference_mode():
        for _ in range(args.warmup):
            flash_attn_func(q, k, v, dropout_p=0.0, causal=True)
        torch.cuda.synchronize()
        out = flash_attn_func(q, k, v, dropout_p=0.0, causal=True)
        torch.cuda.synchronize()

    print(f"checksum={float(out.float().mean().cpu()):.8f}")


if __name__ == "__main__":
    main()
