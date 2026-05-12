#!/usr/bin/env python3
"""Plot Qwen3-8B request-rate serving capacity sweep."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import matplotlib.pyplot as plt


def main() -> None:
    parser = argparse.ArgumentParser(description="Plot Qwen3-8B request-rate sweep")
    parser.add_argument(
        "--input-json",
        type=Path,
        default=Path("benchmark/projects/qwen3_8b_dense/data/request_rate_sweep_512in_256out_single_4090.json"),
    )
    parser.add_argument(
        "--output-png",
        type=Path,
        default=Path("benchmark/projects/qwen3_8b_dense/assets/request_rate_capacity_single_4090.png"),
    )
    args = parser.parse_args()

    payload = json.loads(args.input_json.read_text(encoding="utf-8"))
    rows = sorted(payload["results"], key=lambda x: x["request_rate"])

    x = [r["request_rate"] for r in rows]
    output_tps = [r["output_token_throughput_tok_s"] for r in rows]
    p99_ttft = [r["p99_ttft_ms"] for r in rows]
    p99_itl = [r["p99_itl_ms"] for r in rows]

    plt.rcParams.update(
        {
            "font.size": 11,
            "axes.titlesize": 15,
            "axes.labelsize": 12,
            "legend.fontsize": 10,
            "figure.facecolor": "white",
            "axes.facecolor": "#fbfbf8",
            "axes.edgecolor": "#2f3437",
            "axes.grid": True,
            "grid.color": "#d8d6cf",
            "grid.linewidth": 0.7,
        }
    )

    fig, ax1 = plt.subplots(figsize=(11, 6.2))
    fig.subplots_adjust(right=0.78)

    ax1.axvspan(8, 16, color="#eadfcb", alpha=0.45, zorder=0)
    l1 = ax1.plot(
        x,
        output_tps,
        marker="o",
        linewidth=2.8,
        color="#177245",
        label="Output throughput (tok/s)",
    )
    ax1.set_xlabel("Configured request rate (req/s)")
    ax1.set_ylabel("Output throughput (tok/s)", color="#177245")
    ax1.tick_params(axis="y", labelcolor="#177245")
    ax1.set_xticks(x)
    ax1.set_ylim(0, max(output_tps) * 1.18)

    ax2 = ax1.twinx()
    l2 = ax2.plot(
        x,
        p99_ttft,
        marker="s",
        linewidth=2.5,
        color="#b14d1f",
        label="P99 TTFT (ms)",
    )
    ax2.set_ylabel("P99 TTFT (ms)", color="#b14d1f")
    ax2.tick_params(axis="y", labelcolor="#b14d1f")
    ax2.set_ylim(0, max(p99_ttft) * 1.2)

    ax3 = ax1.twinx()
    ax3.spines["right"].set_position(("axes", 1.14))
    l3 = ax3.plot(
        x,
        p99_itl,
        marker="^",
        linewidth=2.5,
        color="#334f8d",
        label="P99 ITL (ms)",
    )
    ax3.set_ylabel("P99 ITL (ms)", color="#334f8d")
    ax3.tick_params(axis="y", labelcolor="#334f8d")
    ax3.set_ylim(0, max(p99_itl) * 1.35)

    ax1.set_title("Qwen3-8B Serving Capacity vs Request Rate")
    ax1.text(
        8.15,
        max(output_tps) * 1.07,
        "capacity plateau",
        color="#6c5840",
        fontsize=11,
        ha="left",
        va="center",
    )
    ax1.annotate(
        "throughput flattens after ~8 req/s",
        xy=(8, output_tps[x.index(8)]),
        xytext=(5.1, max(output_tps) * 0.82),
        arrowprops={"arrowstyle": "->", "color": "#5f6b60", "linewidth": 1.4},
        color="#33403a",
    )
    ax2.annotate(
        "higher offered load mainly raises queueing latency",
        xy=(16, p99_ttft[-1]),
        xytext=(8.7, max(p99_ttft) * 0.68),
        arrowprops={"arrowstyle": "->", "color": "#8a4325", "linewidth": 1.4},
        color="#7a3a20",
    )

    lines = l1 + l2 + l3
    labels = [line.get_label() for line in lines]
    ax1.legend(lines, labels, loc="upper left", frameon=True, framealpha=0.92)

    caption = (
        "Hardware: single RTX 4090. Dataset: random, input=512, output=256, num_prompts=256, "
        "max_concurrency=32, temperature=0"
    )
    fig.text(0.08, 0.025, caption, color="#525252", fontsize=9)

    args.output_png.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(args.output_png, dpi=220, bbox_inches="tight")
    plt.close(fig)
    print(f"Saved {args.output_png}")


if __name__ == "__main__":
    main()
