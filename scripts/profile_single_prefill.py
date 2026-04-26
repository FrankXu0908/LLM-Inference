#!/usr/bin/env python3
import argparse
import json
from pathlib import Path
from typing import Dict, List

import torch
from torch.profiler import ProfilerActivity, profile
from transformers import AutoModelForCausalLM, AutoTokenizer


ATTN_MODULE_KEYS = ("attn", "attention")
MLP_MODULE_KEYS = ("mlp", "ffn", "feed_forward")

ATTN_OP_KEYS = (
    "attention",
    "attn",
    "scaled_dot_product",
    "flash_attention",
    "flash_attn",
    "sdpa",
)
GEMM_OP_KEYS = (
    "gemm",
    "linear",
    "matmul",
    "addmm",
    "mm",
    "bmm",
    "cublas",
    "cutlass",
)


def classify_module(name: str) -> str:
    lname = name.lower()
    if any(k in lname for k in ATTN_MODULE_KEYS):
        return "attention"
    if any(k in lname for k in MLP_MODULE_KEYS):
        return "mlp"
    return "other"


def classify_op(op_name: str) -> str:
    lname = op_name.lower()
    if any(k in lname for k in ATTN_OP_KEYS):
        return "attention_ops"
    if any(k in lname for k in GEMM_OP_KEYS):
        return "gemm_linear_mlp_ops"
    return "other_ops"


def pick_profile_modules(model: torch.nn.Module) -> Dict[str, torch.nn.Module]:
    picked = {}
    for name, module in model.named_modules():
        if not name:
            continue
        leaf = name.split(".")[-1].lower()
        if leaf in ("self_attn", "attn", "attention", "mlp", "ffn", "feed_forward"):
            picked[name] = module
    return picked


def profile_once(
    model: torch.nn.Module,
    seq_len: int,
    token_id: int,
    device: torch.device,
    profile_modules: Dict[str, torch.nn.Module],
) -> Dict:
    module_times_ms = {"attention": 0.0, "mlp": 0.0, "other": 0.0}
    op_times_us = {"attention_ops": 0.0, "gemm_linear_mlp_ops": 0.0, "other_ops": 0.0}
    handles = []

    def make_pre(name: str):
        def _pre(_module, _inputs):
            start = torch.cuda.Event(enable_timing=True)
            end = torch.cuda.Event(enable_timing=True)
            start.record()
            _module.__profile_cuda_events = (name, start, end)
        return _pre

    def make_post(name: str):
        def _post(module, _inputs, _output):
            _n, start, end = module.__profile_cuda_events
            end.record()
            torch.cuda.synchronize()
            dt = start.elapsed_time(end)
            module_times_ms[classify_module(name)] += dt
            del module.__profile_cuda_events
        return _post

    for name, module in profile_modules.items():
        handles.append(module.register_forward_pre_hook(make_pre(name)))
        handles.append(module.register_forward_hook(make_post(name)))

    input_ids = torch.full((1, seq_len), token_id, dtype=torch.long, device=device)
    attention_mask = torch.ones_like(input_ids, dtype=torch.long)

    with torch.inference_mode():
        _ = model(input_ids=input_ids, attention_mask=attention_mask, use_cache=True)
        torch.cuda.synchronize()

    with profile(
        activities=[ProfilerActivity.CPU, ProfilerActivity.CUDA],
        record_shapes=False,
        profile_memory=False,
        with_stack=False,
    ) as prof:
        with torch.inference_mode():
            _ = model(input_ids=input_ids, attention_mask=attention_mask, use_cache=True)
            torch.cuda.synchronize()

    for evt in prof.key_averages():
        cuda_us = float(getattr(evt, "self_cuda_time_total", 0.0))
        if cuda_us <= 0:
            continue
        op_times_us[classify_op(evt.key)] += cuda_us

    for h in handles:
        h.remove()

    module_total_ms = sum(module_times_ms.values())
    op_total_us = sum(op_times_us.values())
    return {
        "seq_len": seq_len,
        "module_cuda_ms": module_times_ms,
        "module_ratio_pct": {
            k: (v / module_total_ms * 100.0 if module_total_ms > 0 else 0.0)
            for k, v in module_times_ms.items()
        },
        "op_cuda_us": op_times_us,
        "op_ratio_pct": {
            k: (v / op_total_us * 100.0 if op_total_us > 0 else 0.0)
            for k, v in op_times_us.items()
        },
    }


def main() -> None:
    parser = argparse.ArgumentParser(description="Single-request prefill module profiling")
    parser.add_argument(
        "--model",
        type=str,
        default="/home/xuliren/repo/models/Qwen/Qwen3.5-35B-A3B-GPTQ-Int4",
    )
    parser.add_argument(
        "--lengths",
        type=int,
        nargs="+",
        default=[256, 2048, 4096, 8192],
    )
    parser.add_argument(
        "--output-json",
        type=Path,
        default=Path("results/tables/Qwen3.5-35B-A3B-GPTQ-Int4/prefill_module_profile.json"),
    )
    args = parser.parse_args()

    if not torch.cuda.is_available():
        raise RuntimeError("CUDA is not available. Please run with GPU access.")

    torch.backends.cuda.matmul.allow_tf32 = True

    print(f"Loading tokenizer from: {args.model}")
    tokenizer = AutoTokenizer.from_pretrained(args.model, trust_remote_code=True)
    print(f"Loading model from: {args.model}")
    model = AutoModelForCausalLM.from_pretrained(
        args.model,
        trust_remote_code=True,
        device_map="auto",
        torch_dtype="auto",
    )
    model.eval()

    first_param = next(model.parameters())
    device = first_param.device
    print(f"Model first device: {device}")

    token_id = tokenizer.encode("Hello", add_special_tokens=False)[0]
    profile_modules = pick_profile_modules(model)
    print(f"Tracked modules: {len(profile_modules)}")

    results: List[Dict] = []
    for seq_len in args.lengths:
        print(f"Profiling prefill seq_len={seq_len} ...")
        one = profile_once(
            model=model,
            seq_len=seq_len,
            token_id=token_id,
            device=device,
            profile_modules=profile_modules,
        )
        results.append(one)
        print(
            "  module ratio -> "
            f"attention={one['module_ratio_pct']['attention']:.2f}%, "
            f"mlp={one['module_ratio_pct']['mlp']:.2f}%, "
            f"other={one['module_ratio_pct']['other']:.2f}%"
        )

    args.output_json.parent.mkdir(parents=True, exist_ok=True)
    payload = {
        "model": args.model,
        "lengths": args.lengths,
        "results": results,
    }
    with open(args.output_json, "w", encoding="utf-8") as f:
        json.dump(payload, f, indent=2, ensure_ascii=False)
    print(f"Saved profiling result to {args.output_json}")


if __name__ == "__main__":
    main()
