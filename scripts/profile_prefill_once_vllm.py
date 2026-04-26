#!/usr/bin/env python3
import argparse

import torch
from transformers import AutoTokenizer
from vllm import LLM, SamplingParams


def main() -> None:
    parser = argparse.ArgumentParser(description="Run one single-request prefill-oriented vLLM pass")
    parser.add_argument(
        "--model",
        type=str,
        default="/home/xuliren/repo/models/Qwen/Qwen3.5-35B-A3B-GPTQ-Int4",
    )
    parser.add_argument("--input-tokens", type=int, required=True)
    parser.add_argument("--tensor-parallel-size", type=int, default=2)
    parser.add_argument("--max-model-len", type=int, default=8192)
    args = parser.parse_args()

    tokenizer = AutoTokenizer.from_pretrained(args.model, trust_remote_code=True)
    token_id = tokenizer.encode("Hello", add_special_tokens=False)[0]
    token_ids = [token_id] * args.input_tokens
    if tokenizer.bos_token_id is not None and token_ids:
        token_ids[0] = tokenizer.bos_token_id

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

    sampling = SamplingParams(
        temperature=0.0,
        top_p=1.0,
        max_tokens=1,
        ignore_eos=True,
    )

    # Warmup to reduce one-time init noise in the measured run.
    llm.generate([{"prompt_token_ids": token_ids[: min(32, len(token_ids))]}], sampling)
    torch.cuda.synchronize()

    # Measure only this request with nsys --capture-range=cudaProfilerApi.
    cudart = torch.cuda.cudart()
    cudart.cudaProfilerStart()
    llm.generate([{"prompt_token_ids": token_ids}], sampling)
    torch.cuda.synchronize()
    cudart.cudaProfilerStop()


if __name__ == "__main__":
    main()
