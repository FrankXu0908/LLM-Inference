import argparse
import json
from pathlib import Path

import matplotlib.pyplot as plt


def plot_metrics(results_path: Path, output_dir: Path) -> None:
    with open(results_path, "r", encoding="utf-8") as f:
        data = json.load(f)

    output_dir.mkdir(parents=True, exist_ok=True)

    for metric in ["ttft", "latency", "throughput"]:
        plt.figure(figsize=(8, 5))

        for tokens in sorted({d["input_tokens"] for d in data}):
            xs = []
            ys = []
            for d in data:
                if d["input_tokens"] == tokens:
                    xs.append(d["concurrency"])
                    ys.append(d[metric])

            pairs = sorted(zip(xs, ys), key=lambda p: p[0])
            xs_sorted = [p[0] for p in pairs]
            ys_sorted = [p[1] for p in pairs]

            plt.plot(xs_sorted, ys_sorted, marker="o", label=f"{tokens} tokens")

        plt.xlabel("Concurrency")
        plt.ylabel(metric)
        plt.legend()
        plt.title(metric)
        plt.grid(alpha=0.25)
        plt.tight_layout()
        plt.savefig(output_dir / f"{metric}.png", dpi=180)
        plt.close()


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--results-json",
        type=Path,
        default=Path("results/tables/Qwen3.5-35B-A3B-GPTQ-Int4/results.json"),
        help="Path to sweep results JSON",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path("results/figures/Qwen3.5-35B-A3B-GPTQ-Int4"),
        help="Directory to save generated plots",
    )
    return parser.parse_args()


if __name__ == "__main__":
    args = parse_args()
    plot_metrics(args.results_json, args.output_dir)
    print(f"Saved plots to {args.output_dir}")
