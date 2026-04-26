#!/usr/bin/env python3
import argparse
import json
from pathlib import Path

import matplotlib.pyplot as plt


def main() -> None:
    parser = argparse.ArgumentParser(description="Plot prefill operator breakdown vs context length")
    parser.add_argument(
        "--input-json",
        type=Path,
        default=Path("results/tables/Qwen3.5-35B-A3B-GPTQ-Int4/prefill_profile.json"),
    )
    parser.add_argument(
        "--output-png",
        type=Path,
        default=Path("results/figures/Qwen3.5-35B-A3B-GPTQ-Int4/prefill_operator_breakdown.png"),
    )
    args = parser.parse_args()

    with open(args.input_json, "r", encoding="utf-8") as f:
        payload = json.load(f)

    results = sorted(payload["results"], key=lambda x: x["input_tokens"])
    x_labels = [str(r["input_tokens"]) for r in results]
    attention = [r["ratio_pct"]["attention"] for r in results]
    mlp = [r["ratio_pct"]["mlp_gemm"] for r in results]
    others = [r["ratio_pct"]["other"] for r in results]

    fig, ax = plt.subplots(figsize=(9, 5))
    ax.bar(x_labels, attention, label="attention")
    ax.bar(x_labels, mlp, bottom=attention, label="MLP")
    bottom_others = [a + m for a, m in zip(attention, mlp)]
    ax.bar(x_labels, others, bottom=bottom_others, label="others")

    ax.set_xlabel("Context Length")
    ax.set_ylabel("CUDA Time Share (%)")
    ax.set_title("Prefill Operator Breakdown vs Context Length")
    ax.set_ylim(0, 100)
    ax.legend()
    ax.grid(axis="y", alpha=0.25)
    plt.tight_layout()

    args.output_png.parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(args.output_png, dpi=180)
    plt.close(fig)
    print(f"Saved figure to {args.output_png}")


if __name__ == "__main__":
    main()
