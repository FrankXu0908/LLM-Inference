import argparse
import json
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
from model_config import load_model_name


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Plot TP2/DP2/DP2+EP comparison heatmaps")
    parser.add_argument("--model", type=str, default=None)
    parser.add_argument("--model-config", type=str, default="configs/model.yaml")
    parser.add_argument("--tables-root", type=Path, default=Path("results/tables"))
    parser.add_argument("--figures-root", type=Path, default=Path("results/figures"))
    parser.add_argument("--compare-json", type=Path, default=None)
    parser.add_argument("--output-dir", type=Path, default=None)
    return parser.parse_args()


def build_matrix(rows, comp_key, metric_key, concs, toks):
    mat = np.full((len(concs), len(toks)), np.nan, dtype=float)
    c_idx = {c: i for i, c in enumerate(concs)}
    t_idx = {t: i for i, t in enumerate(toks)}
    for r in rows:
        c = r.get("concurrency")
        t = r.get("input_tokens")
        comp = r.get(comp_key) or {}
        v = comp.get(metric_key)
        if c in c_idx and t in t_idx and isinstance(v, (int, float)):
            mat[c_idx[c], t_idx[t]] = float(v)
    return mat


def annotate(ax, mat):
    for i in range(mat.shape[0]):
        for j in range(mat.shape[1]):
            val = mat[i, j]
            if np.isnan(val):
                txt = "NA"
            else:
                txt = f"{val:.1f}%"
            ax.text(j, i, txt, ha="center", va="center", fontsize=8, color="black")


def plot_one(rows, comp_key, title, output_png: Path):
    concs = sorted({int(r["concurrency"]) for r in rows if r.get("concurrency") is not None})
    toks = sorted({int(r["input_tokens"]) for r in rows if r.get("input_tokens") is not None})
    metrics = [
        ("ttft_pct", "TTFT delta %"),
        ("latency_pct", "Latency delta %"),
        ("throughput_pct", "Throughput delta %"),
    ]

    fig, axes = plt.subplots(1, 3, figsize=(16, 4.8))
    cmap = "RdYlGn"
    for ax, (metric_key, metric_title) in zip(axes, metrics):
        mat = build_matrix(rows, comp_key, metric_key, concs, toks)
        im = ax.imshow(mat, aspect="auto", cmap=cmap)
        ax.set_title(metric_title)
        ax.set_xticks(range(len(toks)))
        ax.set_xticklabels([str(t) for t in toks], rotation=0)
        ax.set_yticks(range(len(concs)))
        ax.set_yticklabels([str(c) for c in concs])
        ax.set_xlabel("Input tokens")
        ax.set_ylabel("Concurrency")
        annotate(ax, mat)
        fig.colorbar(im, ax=ax, fraction=0.046, pad=0.04)

    fig.suptitle(title, fontsize=14)
    output_png.parent.mkdir(parents=True, exist_ok=True)
    plt.tight_layout()
    plt.savefig(output_png, dpi=160)
    plt.close(fig)


def main() -> None:
    args = parse_args()
    if args.model is None:
        args.model = load_model_name(args.model_config)
    model_name = Path(args.model).name
    if args.compare_json is None:
        args.compare_json = args.tables_root / model_name / "compare_tp2_dp2_dp2ep.json"
    if args.output_dir is None:
        args.output_dir = args.figures_root / model_name / "compare"

    with open(args.compare_json, "r", encoding="utf-8") as f:
        rows = json.load(f)

    plot_one(
        rows,
        comp_key="delta_dp2_vs_tp2",
        title="DP2 vs TP2 (delta %)",
        output_png=args.output_dir / "dp2_vs_tp2_heatmaps.png",
    )
    plot_one(
        rows,
        comp_key="delta_dp2ep_vs_tp2",
        title="DP2+EP vs TP2 (delta %)",
        output_png=args.output_dir / "dp2ep_vs_tp2_heatmaps.png",
    )
    plot_one(
        rows,
        comp_key="delta_dp2ep_vs_dp2",
        title="DP2+EP vs DP2 (delta %)",
        output_png=args.output_dir / "dp2ep_vs_dp2_heatmaps.png",
    )
    print(f"Saved figures to {args.output_dir}")


if __name__ == "__main__":
    main()
