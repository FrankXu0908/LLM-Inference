#!/usr/bin/env python3
"""Profile one Qwen3-8B BF16 request through vLLM's TRITON_ATTN path.

This script is intended to be launched under Nsight Systems with
`--capture-range=cudaProfilerApi`. It warms up outside the capture window, then
captures exactly one request.
"""

import argparse
import gc
import os
from pathlib import Path
from typing import Any

import torch
import yaml
from transformers import AutoTokenizer
from vllm import LLM, SamplingParams


def load_config(path: Path) -> dict[str, Any]:
    with path.open("r", encoding="utf-8") as f:
        return yaml.safe_load(f) or {}


def make_prompt_ids(tokenizer: AutoTokenizer, target_tokens: int) -> list[int]:
    seed = tokenizer.encode("hello", add_special_tokens=False)
    if not seed:
        raise RuntimeError("Tokenizer returned an empty seed token list.")
    ids = [seed[0]] * target_tokens
    if tokenizer.bos_token_id is not None and ids:
        ids[0] = tokenizer.bos_token_id
    return ids


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Single-request Qwen3-8B BF16 TRITON_ATTN profiling workload"
    )
    parser.add_argument(
        "--model-config",
        type=Path,
        default=Path("configs/qwen3_8b_dense_triton_attn.yaml"),
    )
    parser.add_argument("--input-tokens", type=int, default=8192)
    parser.add_argument("--output-tokens", type=int, default=1)
    parser.add_argument(
        "--phase",
        choices=["prefill", "decode"],
        default="prefill",
        help="Label only. Use output_tokens=1 for prefill-oriented profiling.",
    )
    parser.add_argument("--gpu-memory-utilization", type=float, default=None)
    parser.add_argument("--max-num-seqs", type=int, default=None)
    parser.add_argument("--enforce-eager", action="store_true")
    parser.add_argument(
        "--warmup-tokens",
        type=int,
        default=128,
        help="Warmup prompt length. Set to 0 for Nsight Compute kernel filtering.",
    )
    args = parser.parse_args()

    cfg = load_config(args.model_config)
    model = cfg["model"]
    dtype = cfg.get("dtype", "bfloat16")
    attention_backend = cfg.get("attention_backend", "TRITON_ATTN")
    max_model_len = int(cfg.get("max_model_len", 10000))
    gpu_memory_utilization = float(
        args.gpu_memory_utilization
        if args.gpu_memory_utilization is not None
        else cfg.get("gpu_memory_utilization", 0.95)
    )
    max_num_seqs = int(
        args.max_num_seqs
        if args.max_num_seqs is not None
        else cfg.get("max_num_seqs", 1)
    )
    enforce_eager = bool(args.enforce_eager or cfg.get("enforce_eager", False))

    tokenizer = AutoTokenizer.from_pretrained(model, trust_remote_code=True)
    prompt_ids = make_prompt_ids(tokenizer, args.input_tokens)

    llm = LLM(
        model=model,
        dtype=dtype,
        tensor_parallel_size=int(cfg.get("tensor_parallel_size", 1)),
        max_model_len=max_model_len,
        gpu_memory_utilization=gpu_memory_utilization,
        trust_remote_code=True,
        enforce_eager=enforce_eager,
        enable_prefix_caching=bool(cfg.get("enable_prefix_caching", True)),
        enable_chunked_prefill=bool(cfg.get("enable_chunked_prefill", True)),
        max_num_seqs=max_num_seqs,
        attention_backend=attention_backend,
    )

    warmup_tokens = min(args.warmup_tokens, args.input_tokens)
    warmup_prompt = [{"prompt_token_ids": prompt_ids[:warmup_tokens]}]
    measured_prompt = [{"prompt_token_ids": prompt_ids}]
    warmup_sampling = SamplingParams(
        temperature=0.0,
        top_p=1.0,
        max_tokens=1,
        ignore_eos=True,
    )
    measured_sampling = SamplingParams(
        temperature=0.0,
        top_p=1.0,
        max_tokens=args.output_tokens,
        ignore_eos=True,
    )

    try:
        if warmup_tokens > 0:
            llm.generate(warmup_prompt, warmup_sampling)
            torch.cuda.synchronize()

        torch.cuda.nvtx.range_push(
            f"qwen3_8b_triton_{args.phase}_in{args.input_tokens}_out{args.output_tokens}"
        )
        cudart = torch.cuda.cudart()
        cudart.cudaProfilerStart()
        llm.generate(measured_prompt, measured_sampling)
        torch.cuda.synchronize()
        cudart.cudaProfilerStop()
        torch.cuda.nvtx.range_pop()
    finally:
        engine = getattr(llm, "llm_engine", None)
        if engine is not None:
            try:
                engine.shutdown()
            except Exception:
                pass
        del llm
        gc.collect()

    os._exit(0)


if __name__ == "__main__":
    main()
