#!/usr/bin/env python3
import argparse
import gc
import os

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
    parser = argparse.ArgumentParser(description="Run one decode-oriented vLLM pass")
    parser.add_argument(
        "--model",
        type=str,
        default="/home/xuliren/repo/models/Qwen/Qwen3.5-35B-A3B-GPTQ-Int4",
    )
    parser.add_argument("--tensor-parallel-size", type=int, default=2)
    parser.add_argument("--max-model-len", type=int, default=8192)
    parser.add_argument("--input-tokens", type=int, default=16)
    parser.add_argument("--new-tokens", type=int, default=256)
    args = parser.parse_args()

    tokenizer = AutoTokenizer.from_pretrained(args.model, trust_remote_code=True)
    prompt_ids = make_prompt(tokenizer, args.input_tokens)

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

    # Warmup outside nsys range.
    llm.generate(
        [{"prompt_token_ids": prompt_ids}],
        SamplingParams(max_tokens=1, temperature=0.0, top_p=1.0, ignore_eos=True),
    )
    torch.cuda.synchronize()

    cudart = torch.cuda.cudart()
    try:
        cudart.cudaProfilerStart()
        llm.generate(
            [{"prompt_token_ids": prompt_ids}],
            SamplingParams(
                max_tokens=args.new_tokens,
                temperature=0.0,
                top_p=1.0,
                ignore_eos=True,
            ),
        )
        torch.cuda.synchronize()
    finally:
        # Ensure capture range closes and vLLM workers terminate cleanly so
        # nsys can finalize .nsys-rep instead of leaving only .qdstrm.
        try:
            cudart.cudaProfilerStop()
        except Exception:
            pass
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
