import json
import re
import argparse
import subprocess
from pathlib import Path
from model_config import load_model_name

def parse_args():
    parser = argparse.ArgumentParser(description="Sweep benchmark_vllm across concurrency and input tokens")
    parser.add_argument("--model", type=str, default=None)
    parser.add_argument("--model-config", type=str, default="configs/model.yaml")
    parser.add_argument("--base-url", type=str, default="http://127.0.0.1:8000/v1")
    parser.add_argument(
        "--parallel-mode",
        type=str,
        choices=["tp2", "dp2", "dp2_ep", "custom"],
        default="custom",
    )
    parser.add_argument("--num-requests", type=int, default=32)
    parser.add_argument("--prompt-type", type=str, choices=["short", "medium", "long", "rag"], default="long")
    parser.add_argument("--max-tokens", type=int, default=128)
    parser.add_argument("--timeout", type=float, default=300.0)
    parser.add_argument("--results-root", type=Path, default=Path("results/tables"))
    return parser.parse_args()


def run_benchmark(args, concurrency, input_tokens):
    cmd = [
        "python",
        "scripts/benchmark_vllm.py",
        "--base-url",
        args.base_url,
        "--model",
        args.model,
        "--parallel-mode",
        args.parallel_mode,
        "--concurrency",
        str(concurrency),
        "--num-requests",
        str(args.num_requests),
        "--prompt-type",
        args.prompt_type,
        "--input-tokens",
        str(input_tokens),
        "--max-tokens",
        str(args.max_tokens),
        "--timeout",
        str(args.timeout),
    ]

    output = subprocess.check_output(cmd, text=True)

    match = re.search(r"=== Benchmark Summary ===\n({.*})", output, re.S)
    data = json.loads(match.group(1))

    return {
        "concurrency": concurrency,
        "input_tokens": input_tokens,
        "ttft": data["ttft_mean"],
        "latency": data["latency_mean"],
        "throughput": data["aggregate_output_tps"],
    }


def main():
    args = parse_args()
    if args.model is None:
        args.model = load_model_name(args.model_config)
    model_name = Path(args.model).name
    output_dir = args.results_root / model_name / args.parallel_mode
    results_path = output_dir / "results.json"
    concurrency_list = [4, 8, 16, 32]
    input_tokens_list = [256, 2048, 4096, 8192]
    results = []

    for c in concurrency_list:
        for t in input_tokens_list:
            print(f"Running mode={args.parallel_mode}, c={c}, tokens={t}")
            res = run_benchmark(args, c, t)
            results.append(res)

    output_dir.mkdir(parents=True, exist_ok=True)
    with open(results_path, "w", encoding="utf-8") as f:
        json.dump(results, f, indent=2, ensure_ascii=False)

    print(f"Saved sweep results to {results_path}")


if __name__ == "__main__":
    main()
