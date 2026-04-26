#!/usr/bin/env python3
import argparse
import json
from pathlib import Path
from typing import Dict, List

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer


def build_exact_length_inputs(
    tokenizer: AutoTokenizer, model_device: torch.device, seq_len: int
) -> Dict[str, torch.Tensor]:
    token_ids = tokenizer.encode("hello", add_special_tokens=False)
    if not token_ids:
        raise RuntimeError("Tokenizer returned empty token ids for seed text.")
    seed_id = token_ids[0]

    input_ids = torch.full(
        (1, seq_len),
        seed_id,
        dtype=torch.long,
        device=model_device,
    )
    attention_mask = torch.ones_like(input_ids, dtype=torch.long)
    return {"input_ids": input_ids, "attention_mask": attention_mask}


def top_ops_from_profiler(prof: torch.profiler.profile, top_k: int) -> List[Dict]:
    rows = []
    for evt in prof.key_averages():
        rows.append(
            {
                "op": evt.key,
                "count": int(evt.count),
                "cuda_time_total_us": float(getattr(evt, "cuda_time_total", 0.0)),
                "self_cuda_time_total_us": float(
                    getattr(evt, "self_cuda_time_total", 0.0)
                ),
                "cpu_time_total_us": float(getattr(evt, "cpu_time_total", 0.0)),
                "self_cpu_time_total_us": float(
                    getattr(evt, "self_cpu_time_total", 0.0)
                ),
            }
        )
    rows.sort(key=lambda x: x["cuda_time_total_us"], reverse=True)
    return rows[:top_k]


def run_prefill_forward(model, inputs: Dict[str, torch.Tensor]) -> None:
    backbone = model.model if hasattr(model, "model") else model
    _ = backbone(
        input_ids=inputs["input_ids"],
        attention_mask=inputs["attention_mask"],
        use_cache=False,
        return_dict=False,
    )


def run_one_length(
    model,
    tokenizer,
    seq_len: int,
    max_new_tokens: int,
    top_k: int,
    out_dir: Path,
) -> Dict:
    model_device = next(model.parameters()).device
    inputs = build_exact_length_inputs(tokenizer, model_device, seq_len)

    with torch.no_grad():
        run_prefill_forward(model, inputs)
        torch.cuda.synchronize()

    with torch.no_grad():
        with torch.profiler.profile(
            activities=[
                torch.profiler.ProfilerActivity.CPU,
                torch.profiler.ProfilerActivity.CUDA,
            ],
            record_shapes=True,
            with_stack=False,
            profile_memory=True,
        ) as prof:
            run_prefill_forward(model, inputs)
            torch.cuda.synchronize()

    table_txt = prof.key_averages().table(sort_by="cuda_time_total", row_limit=top_k)
    top_ops = top_ops_from_profiler(prof, top_k=top_k)

    trace_path = out_dir / f"trace_len_{seq_len}.json"
    prof.export_chrome_trace(str(trace_path))

    table_path = out_dir / f"top_ops_len_{seq_len}.txt"
    table_path.write_text(table_txt, encoding="utf-8")

    json_path = out_dir / f"top_ops_len_{seq_len}.json"
    json_path.write_text(
        json.dumps(
            {
                "seq_len": seq_len,
                "max_new_tokens": max_new_tokens,
                "top_ops": top_ops,
            },
            indent=2,
            ensure_ascii=False,
        ),
        encoding="utf-8",
    )

    return {
        "seq_len": seq_len,
        "top_ops_json": str(json_path),
        "top_ops_table": str(table_path),
        "trace_json": str(trace_path),
    }


def main() -> None:
    parser = argparse.ArgumentParser(description="Profile transformers top CUDA ops")
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
    parser.add_argument("--max-new-tokens", type=int, default=1)
    parser.add_argument("--top-k", type=int, default=30)
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path("results/tables/Qwen3.5-35B-A3B-GPTQ-Int4/transformers_top_ops"),
    )
    args = parser.parse_args()

    if not torch.cuda.is_available():
        raise RuntimeError("CUDA is not available.")

    args.output_dir.mkdir(parents=True, exist_ok=True)

    tokenizer = AutoTokenizer.from_pretrained(args.model, trust_remote_code=True)
    model = AutoModelForCausalLM.from_pretrained(
        args.model,
        trust_remote_code=True,
        device_map="auto",
    )
    model.eval()

    summary = {
        "model": args.model,
        "lengths": args.lengths,
        "max_new_tokens": args.max_new_tokens,
        "results": [],
    }
    for seq_len in args.lengths:
        print(f"Profiling length={seq_len}")
        one = run_one_length(
            model=model,
            tokenizer=tokenizer,
            seq_len=seq_len,
            max_new_tokens=args.max_new_tokens,
            top_k=args.top_k,
            out_dir=args.output_dir,
        )
        summary["results"].append(one)

    summary_path = args.output_dir / "summary.json"
    summary_path.write_text(
        json.dumps(summary, indent=2, ensure_ascii=False), encoding="utf-8"
    )
    print(f"Saved summary to {summary_path}")


if __name__ == "__main__":
    main()
