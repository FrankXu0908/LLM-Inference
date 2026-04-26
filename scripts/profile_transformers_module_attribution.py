#!/usr/bin/env python3
import argparse
import json
import re
from collections import defaultdict
from pathlib import Path
from typing import Dict, List, Tuple

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer


LAYER_RE = re.compile(r"model\.layers\.(\d+)\.")


def build_exact_length_inputs(
    tokenizer: AutoTokenizer, device: torch.device, seq_len: int
) -> Dict[str, torch.Tensor]:
    token_ids = tokenizer.encode("hello", add_special_tokens=False)
    if not token_ids:
        raise RuntimeError("Tokenizer returned empty token ids for seed text.")
    seed_id = token_ids[0]
    input_ids = torch.full((1, seq_len), seed_id, dtype=torch.long, device=device)
    attention_mask = torch.ones_like(input_ids, dtype=torch.long)
    return {"input_ids": input_ids, "attention_mask": attention_mask}


def extract_layer_idx(name: str) -> str:
    m = LAYER_RE.search(name)
    return m.group(1) if m else "global"


def parse_layer_slice(spec: str | None, total_layers: int | None = None) -> set[int] | None:
    if not spec:
        return None
    if ":" not in spec:
        raise ValueError(f"Invalid layer slice '{spec}', expected format like 0:3")
    s, e = spec.split(":", 1)
    start = int(s) if s else 0
    end = int(e) if e else (total_layers if total_layers is not None else start)
    if end < start:
        raise ValueError(f"Invalid layer slice '{spec}': end < start")
    return set(range(start, end))


def categorize_module(name: str, module: torch.nn.Module) -> str | None:
    cls = module.__class__.__name__.lower()
    n = name.lower()

    if n.endswith("self_attn") or n.endswith("linear_attn"):
        return "attention_like"

    if n.endswith("mlp.gate") or n.endswith("mlp.shared_expert_gate"):
        return "router_dispatch"

    if n.endswith("mlp.experts") or n.endswith("mlp.shared_expert"):
        return "mlp_experts"

    # Dense MLP layers that are not sparse MoE blocks.
    if n.endswith("mlp") and "moemlp" in cls:
        return "mlp_experts"

    return None


def run_prefill(backbone: torch.nn.Module, inputs: Dict[str, torch.Tensor]) -> None:
    _ = backbone(
        input_ids=inputs["input_ids"],
        attention_mask=inputs["attention_mask"],
        use_cache=False,
        return_dict=False,
    )


def run_prefill_decode1(backbone: torch.nn.Module, inputs: Dict[str, torch.Tensor]) -> None:
    out = backbone(
        input_ids=inputs["input_ids"],
        attention_mask=inputs["attention_mask"],
        use_cache=True,
        return_dict=True,
    )
    next_token = inputs["input_ids"][:, -1:]
    next_mask = torch.ones_like(next_token, dtype=inputs["attention_mask"].dtype)
    _ = backbone(
        input_ids=next_token,
        attention_mask=next_mask,
        past_key_values=out.past_key_values,
        use_cache=True,
        return_dict=False,
    )


def profile_one_length(
    model: torch.nn.Module,
    tokenizer: AutoTokenizer,
    seq_len: int,
    mode: str,
    warmup_runs: int,
    selected_layers: set[int] | None,
) -> Dict:
    backbone = model.model if hasattr(model, "model") else model
    device = next(model.parameters()).device
    inputs = build_exact_length_inputs(tokenizer, device, seq_len)

    module_specs: List[Tuple[str, str, torch.nn.Module]] = []
    for name, module in backbone.named_modules():
        layer_idx = extract_layer_idx(name)
        if selected_layers is not None and layer_idx != "global":
            if int(layer_idx) not in selected_layers:
                continue
        category = categorize_module(name, module)
        if category is not None:
            module_specs.append((name, category, module))

    events: List[Tuple[str, str, torch.cuda.Event, torch.cuda.Event]] = []
    in_flight: Dict[int, Tuple[str, str, torch.cuda.Event]] = {}
    handles = []

    def make_pre(name: str, category: str):
        def _pre(module: torch.nn.Module, _inputs):
            start = torch.cuda.Event(enable_timing=True)
            start.record()
            in_flight[id(module)] = (name, category, start)
            torch.cuda.nvtx.range_push(f"{category}:{name}")
        return _pre

    def make_post():
        def _post(module: torch.nn.Module, _inputs, _output):
            meta = in_flight.pop(id(module), None)
            if meta is None:
                return
            name, category, start = meta
            end = torch.cuda.Event(enable_timing=True)
            end.record()
            events.append((name, category, start, end))
            torch.cuda.nvtx.range_pop()
        return _post

    for name, category, module in module_specs:
        handles.append(module.register_forward_pre_hook(make_pre(name, category)))
        handles.append(module.register_forward_hook(make_post()))

    for _ in range(warmup_runs):
        with torch.no_grad():
            run_prefill(backbone, inputs)
            torch.cuda.synchronize()

    total_start = torch.cuda.Event(enable_timing=True)
    total_end = torch.cuda.Event(enable_timing=True)
    with torch.no_grad():
        total_start.record()
        if mode == "prefill":
            run_prefill(backbone, inputs)
        else:
            run_prefill_decode1(backbone, inputs)
        total_end.record()
        torch.cuda.synchronize()

    for h in handles:
        h.remove()

    category_ms = defaultdict(float)
    layer_category_ms: Dict[str, Dict[str, float]] = defaultdict(lambda: defaultdict(float))
    modules_ms = []

    for name, category, start, end in events:
        ms = float(start.elapsed_time(end))
        category_ms[category] += ms
        layer_idx = extract_layer_idx(name)
        layer_category_ms[layer_idx][category] += ms
        modules_ms.append({"module": name, "category": category, "cuda_ms": ms})

    total_ms = float(total_start.elapsed_time(total_end))
    known_ms = (
        category_ms["attention_like"]
        + category_ms["mlp_experts"]
        + category_ms["router_dispatch"]
    )
    other_ms = max(0.0, total_ms - known_ms)
    category_ms["other"] += other_ms

    ratio = {
        k: (v / total_ms * 100.0 if total_ms > 0 else 0.0)
        for k, v in {
            "attention_like": category_ms["attention_like"],
            "mlp_experts": category_ms["mlp_experts"],
            "router_dispatch": category_ms["router_dispatch"],
            "other": category_ms["other"],
        }.items()
    }

    layer_summary = []
    for layer_idx in sorted(
        [k for k in layer_category_ms.keys() if k != "global"], key=lambda x: int(x)
    ):
        item = {
            "layer": int(layer_idx),
            "attention_like_ms": float(layer_category_ms[layer_idx].get("attention_like", 0.0)),
            "mlp_experts_ms": float(layer_category_ms[layer_idx].get("mlp_experts", 0.0)),
            "router_dispatch_ms": float(layer_category_ms[layer_idx].get("router_dispatch", 0.0)),
        }
        layer_summary.append(item)

    return {
        "seq_len": seq_len,
        "mode": mode,
        "total_cuda_ms": total_ms,
        "category_cuda_ms": {
            "attention_like": float(category_ms["attention_like"]),
            "mlp_experts": float(category_ms["mlp_experts"]),
            "router_dispatch": float(category_ms["router_dispatch"]),
            "other": float(category_ms["other"]),
        },
        "category_ratio_pct": ratio,
        "layer_summary": layer_summary,
        "module_count_profiled": len(module_specs),
        "module_events": modules_ms,
    }


def main() -> None:
    parser = argparse.ArgumentParser(description="Transformers module-level attribution for Qwen3.5-MoE")
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
        "--mode",
        type=str,
        choices=["prefill", "prefill_decode1"],
        default="prefill",
    )
    parser.add_argument(
        "--output-json",
        type=Path,
        default=Path("results/tables/Qwen3.5-35B-A3B-GPTQ-Int4/transformers_module_attribution.json"),
    )
    parser.add_argument(
        "--warmup-runs",
        type=int,
        default=0,
        help="Warmup forward passes before measured run (0 is fastest).",
    )
    parser.add_argument(
        "--quick",
        action="store_true",
        help="Quick mode: run only 2048 prefill with no warmup.",
    )
    parser.add_argument(
        "--layer-slice",
        type=str,
        default=None,
        help="Only attribute modules in this layer slice, e.g. 0:3",
    )
    parser.add_argument(
        "--truncate-to-layer-slice",
        action="store_true",
        help="Execute only the selected layer slice (much faster, but partial model).",
    )
    args = parser.parse_args()

    if args.quick:
        args.lengths = [2048]
        args.mode = "prefill"
        args.warmup_runs = 0

    if not torch.cuda.is_available():
        raise RuntimeError("CUDA is not available. Please run with GPU access.")

    print("Loading tokenizer...", flush=True)
    tokenizer = AutoTokenizer.from_pretrained(args.model, trust_remote_code=True)
    print("Loading model...", flush=True)
    model = AutoModelForCausalLM.from_pretrained(
        args.model,
        trust_remote_code=True,
        device_map="auto",
        torch_dtype=torch.bfloat16,
    )
    model.eval()

    backbone = model.model if hasattr(model, "model") else model
    cfg = getattr(model, "config", None)
    text_cfg = getattr(cfg, "text_config", None) if cfg is not None else None
    total_layers = getattr(text_cfg, "num_hidden_layers", None)
    selected_layers = parse_layer_slice(args.layer_slice, total_layers=total_layers)
    if text_cfg is not None:
        layer_types = getattr(text_cfg, "layer_types", None)
        if isinstance(layer_types, list):
            print(
                f"Layer types head: {layer_types[:8]} (total={len(layer_types)})",
                flush=True,
            )

    if args.truncate_to_layer_slice:
        if selected_layers is None:
            raise ValueError("--truncate-to-layer-slice requires --layer-slice")
        if not hasattr(backbone, "layers"):
            raise ValueError("Model backbone does not expose .layers; cannot truncate.")
        ordered = sorted(selected_layers)
        layers = backbone.layers
        sliced = torch.nn.ModuleList([layers[i] for i in ordered])
        backbone.layers = sliced
        if text_cfg is not None and hasattr(text_cfg, "num_hidden_layers"):
            text_cfg.num_hidden_layers = len(ordered)
        if text_cfg is not None and hasattr(text_cfg, "layer_types") and isinstance(
            text_cfg.layer_types, list
        ):
            text_cfg.layer_types = [text_cfg.layer_types[i] for i in ordered]
        print(
            f"Truncated execution layers to {ordered} (count={len(ordered)})",
            flush=True,
        )

    print("Model loaded. Start profiling...", flush=True)

    results = []
    for seq_len in args.lengths:
        print(
            f"Profiling seq_len={seq_len} mode={args.mode} warmup_runs={args.warmup_runs}",
            flush=True,
        )
        one = profile_one_length(
            model,
            tokenizer,
            seq_len=seq_len,
            mode=args.mode,
            warmup_runs=args.warmup_runs,
            selected_layers=selected_layers,
        )
        results.append(one)
        r = one["category_ratio_pct"]
        print(
            f"  attention={r['attention_like']:.2f}% "
            f"mlp={r['mlp_experts']:.2f}% "
            f"router={r['router_dispatch']:.2f}% "
            f"other={r['other']:.2f}%",
            flush=True,
        )
        torch.cuda.empty_cache()

    payload = {
        "model": args.model,
        "mode": args.mode,
        "lengths": args.lengths,
        "results": results,
    }
    args.output_json.parent.mkdir(parents=True, exist_ok=True)
    with open(args.output_json, "w", encoding="utf-8") as f:
        json.dump(payload, f, indent=2, ensure_ascii=False)
    print(f"Saved attribution to {args.output_json}", flush=True)


if __name__ == "__main__":
    main()
