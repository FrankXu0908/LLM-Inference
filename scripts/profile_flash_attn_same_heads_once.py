#!/usr/bin/env python3
"""One same-head FA1/FA2 CUDA forward call for Nsight Compute profiling."""

from __future__ import annotations

import argparse
import sys

import torch


def load_backend(backend: str, flash_attn_source: str | None, fa2_interface: str):
    if flash_attn_source:
        sys.path.insert(0, flash_attn_source)

    if backend == "fa2":
        if fa2_interface == "dense":
            from flash_attn import flash_attn_func

            def run(q, k, v, batch, seq_len):
                return flash_attn_func(
                    q.view(batch, seq_len, q.shape[1], q.shape[2]),
                    k.view(batch, seq_len, k.shape[1], k.shape[2]),
                    v.view(batch, seq_len, v.shape[1], v.shape[2]),
                    dropout_p=0.0,
                    causal=True,
                )

            return run

        from flash_attn import flash_attn_varlen_func

        def run(q, k, v, batch, seq_len):
            cu = torch.arange(0, (batch + 1) * seq_len, seq_len, device=q.device, dtype=torch.int32)
            return flash_attn_varlen_func(q, k, v, cu, cu, seq_len, seq_len, dropout_p=0.0, causal=True)

        return run

    if backend == "fa1":
        from flash_attn.flash_attn_interface import flash_attn_unpadded_func

        def run(q, k, v, batch, seq_len):
            cu = torch.arange(0, (batch + 1) * seq_len, seq_len, device=q.device, dtype=torch.int32)
            return flash_attn_unpadded_func(q, k, v, cu, cu, seq_len, seq_len, 0.0, causal=True)

        return run

    raise ValueError(f"Unsupported backend: {backend}")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="One-shot same-head FA1/FA2 CUDA forward")
    parser.add_argument("--backend", choices=["fa1", "fa2"], required=True)
    parser.add_argument("--fa2-interface", choices=["varlen", "dense"], default="varlen")
    parser.add_argument("--flash-attn-source", default=None)
    parser.add_argument("--batch", type=int, default=1)
    parser.add_argument("--seq-len", type=int, default=8192)
    parser.add_argument("--heads", type=int, default=32)
    parser.add_argument("--head-dim", type=int, default=128)
    parser.add_argument("--dtype", choices=["bf16", "fp16"], default="bf16")
    parser.add_argument("--warmup", type=int, default=3)
    parser.add_argument("--seed", type=int, default=42)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    if not torch.cuda.is_available():
        raise RuntimeError("CUDA is not available in this environment.")

    dtype = torch.bfloat16 if args.dtype == "bf16" else torch.float16
    fn = load_backend(args.backend, args.flash_attn_source, args.fa2_interface)
    torch.manual_seed(args.seed)

    total = args.batch * args.seq_len
    q = torch.randn((total, args.heads, args.head_dim), device="cuda", dtype=dtype)
    k = torch.randn((total, args.heads, args.head_dim), device="cuda", dtype=dtype)
    v = torch.randn((total, args.heads, args.head_dim), device="cuda", dtype=dtype)

    print(
        f"{args.backend.upper()} same-head one-shot: "
        f"batch={args.batch} seq={args.seq_len} heads={args.heads} "
        f"head_dim={args.head_dim} dtype={args.dtype} "
        f"fa2_interface={args.fa2_interface if args.backend == 'fa2' else 'n/a'}"
    )
    with torch.inference_mode():
        for _ in range(args.warmup):
            fn(q, k, v, args.batch, args.seq_len)
        torch.cuda.synchronize()
        out = fn(q, k, v, args.batch, args.seq_len)
        torch.cuda.synchronize()
    print(f"checksum={float(out.float().mean().cpu()):.8f}")


if __name__ == "__main__":
    main()
