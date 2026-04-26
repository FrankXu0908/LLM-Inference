#!/usr/bin/env python3
import argparse

import torch
from transformers import AutoTokenizer
from vllm import LLM, SamplingParams


def make_prompt(tokenizer: AutoTokenizer, target_tokens: int) -> list[int]:
    seed = tokenizer.encode("hello", add_special_tokens=False)
    if not seed:
        raise RuntimeError("Tokenizer returned empty seed tokens.")
    token_id = seed[0]
    ids = [token_id] * target_tokens
    if tokenizer.bos_token_id is not None and ids:
        ids[0] = tokenizer.bos_token_id
    return ids


def main() -> None:
    parser = argparse.ArgumentParser(description="Capture execution path with NVTX phases for Nsight Systems")
    parser.add_argument(
        "--model",
        type=str,
        default="/home/xuliren/repo/models/Qwen/Qwen3.5-35B-A3B-GPTQ-Int4",
    )
    parser.add_argument("--tensor-parallel-size", type=int, default=2)
    parser.add_argument("--max-model-len", type=int, default=8192)
    parser.add_argument("--prefill-input-tokens", type=int, default=4096)
    parser.add_argument("--prefill-new-tokens", type=int, default=1)
    parser.add_argument("--decode-input-tokens", type=int, default=16)
    parser.add_argument("--decode-new-tokens", type=int, default=256)
    args = parser.parse_args()

    tokenizer = AutoTokenizer.from_pretrained(args.model, trust_remote_code=True)
    prefill_ids = make_prompt(tokenizer, args.prefill_input_tokens)
    decode_ids = make_prompt(tokenizer, args.decode_input_tokens)

    llm = LLM(
        model=args.model,
        tensor_parallel_size=args.tensor_parallel_size,
        max_model_len=args.max_model_len,
        trust_remote_code=True,
        enforce_eager=True,
        enable_prefix_caching=False,
        max_num_seqs=1,
        gpu_memory_utilization=0.75,
    )

    # Warmup outside measurement range.
    warmup_params = SamplingParams(max_tokens=1, temperature=0.0, top_p=1.0, ignore_eos=True)
    llm.generate([{"prompt_token_ids": prefill_ids[: min(32, len(prefill_ids))]}], warmup_params)
    torch.cuda.synchronize()

    # Start nsys capture range.
    cudart = torch.cuda.cudart()
    cudart.cudaProfilerStart()

    # Phase 1: prefill-dominant.
    torch.cuda.nvtx.range_push("PHASE_PREFILL")
    llm.generate(
        [{"prompt_token_ids": prefill_ids}],
        SamplingParams(
            max_tokens=args.prefill_new_tokens,
            temperature=0.0,
            top_p=1.0,
            ignore_eos=True,
        ),
    )
    torch.cuda.synchronize()
    torch.cuda.nvtx.range_pop()

    # Phase 2: decode-dominant.
    torch.cuda.nvtx.range_push("PHASE_DECODE")
    llm.generate(
        [{"prompt_token_ids": decode_ids}],
        SamplingParams(
            max_tokens=args.decode_new_tokens,
            temperature=0.0,
            top_p=1.0,
            ignore_eos=True,
        ),
    )
    torch.cuda.synchronize()
    torch.cuda.nvtx.range_pop()

    cudart.cudaProfilerStop()


if __name__ == "__main__":
    main()
