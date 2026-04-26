import argparse
import json
import statistics
from pathlib import Path

import matplotlib.pyplot as plt
from model_config import load_model_name


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Compare retest runs for one anomaly point")
    parser.add_argument("--model", type=str, default=None)
    parser.add_argument("--model-config", type=str, default="configs/model.yaml")
    parser.add_argument("--tables-root", type=Path, default=Path("results/tables"))
    parser.add_argument("--figures-root", type=Path, default=Path("results/figures"))
    parser.add_argument(
        "--root",
        type=Path,
        default=None,
    )
    parser.add_argument("--mode-a", type=str, default="dp2")
    parser.add_argument("--mode-b", type=str, default="dp2_ep")
    parser.add_argument("--first-compare-json", type=Path, default=None)
    parser.add_argument("--concurrency", type=int, default=16)
    parser.add_argument("--input-tokens", type=int, default=256)
    parser.add_argument(
        "--output-json",
        type=Path,
        default=None,
    )
    parser.add_argument(
        "--output-fig",
        type=Path,
        default=None,
    )
    return parser.parse_args()


def load_mode_runs(root: Path, mode: str) -> list[dict]:
    mode_dir = root / mode
    runs = []
    for p in sorted(mode_dir.glob("run*.json")):
        with open(p, "r", encoding="utf-8") as f:
            payload = json.load(f)
        s = payload.get("summary", {})
        runs.append(
            {
                "file": str(p),
                "ttft_mean": float(s.get("ttft_mean", 0.0)),
                "latency_mean": float(s.get("latency_mean", 0.0)),
                "throughput": float(s.get("aggregate_output_tps", 0.0)),
            }
        )
    return runs


def summarize(runs: list[dict]) -> dict:
    if not runs:
        return {"count": 0}
    tt = [x["ttft_mean"] for x in runs]
    la = [x["latency_mean"] for x in runs]
    th = [x["throughput"] for x in runs]
    return {
        "count": len(runs),
        "ttft_median": statistics.median(tt),
        "latency_median": statistics.median(la),
        "throughput_median": statistics.median(th),
        "ttft_mean": statistics.mean(tt),
        "latency_mean": statistics.mean(la),
        "throughput_mean": statistics.mean(th),
    }


def get_round1_point(compare_json: Path, c: int, t: int) -> dict | None:
    if not compare_json.exists():
        return None
    with open(compare_json, "r", encoding="utf-8") as f:
        rows = json.load(f)
    for r in rows:
        if r.get("concurrency") == c and r.get("input_tokens") == t:
            return r
    return None


def main() -> None:
    args = parse_args()
    if args.model is None:
        args.model = load_model_name(args.model_config)
    model_name = Path(args.model).name
    if args.root is None:
        args.root = args.tables_root / model_name / f"retest_c{args.concurrency}_in{args.input_tokens}"
    if args.first_compare_json is None:
        args.first_compare_json = args.tables_root / model_name / "compare_tp2_dp2_dp2ep.json"
    if args.output_json is None:
        args.output_json = args.root / "retest_compare_summary.json"
    if args.output_fig is None:
        args.output_fig = args.figures_root / model_name / "compare" / f"retest_c{args.concurrency}_in{args.input_tokens}_round1_vs_round2.png"

    runs_a = load_mode_runs(args.root, args.mode_a)
    runs_b = load_mode_runs(args.root, args.mode_b)
    sum_a = summarize(runs_a)
    sum_b = summarize(runs_b)
    round1 = get_round1_point(args.first_compare_json, args.concurrency, args.input_tokens)

    out = {
        "retest_point": {
            "concurrency": args.concurrency,
            "input_tokens": args.input_tokens,
        },
        "round2_retest": {
            args.mode_a: {"runs": runs_a, "summary": sum_a},
            args.mode_b: {"runs": runs_b, "summary": sum_b},
        },
        "round1_reference": round1,
    }

    if sum_a.get("count", 0) > 0 and sum_b.get("count", 0) > 0:
        out["round2_delta_mode_b_vs_mode_a_pct"] = {
            "ttft_pct": (sum_b["ttft_median"] - sum_a["ttft_median"]) / max(1e-9, sum_a["ttft_median"]) * 100.0,
            "latency_pct": (sum_b["latency_median"] - sum_a["latency_median"]) / max(1e-9, sum_a["latency_median"]) * 100.0,
            "throughput_pct": (sum_b["throughput_median"] - sum_a["throughput_median"]) / max(1e-9, sum_a["throughput_median"]) * 100.0,
        }

    args.output_json.parent.mkdir(parents=True, exist_ok=True)
    with open(args.output_json, "w", encoding="utf-8") as f:
        json.dump(out, f, indent=2, ensure_ascii=False)

    # Plot: round1 raw point vs round2 median (mode_a vs mode_b)
    if round1 and sum_a.get("count", 0) > 0 and sum_b.get("count", 0) > 0:
        m = ["ttft", "latency", "throughput"]
        round1_a = [
            float(round1.get(args.mode_a, {}).get("ttft", 0.0)),
            float(round1.get(args.mode_a, {}).get("latency", 0.0)),
            float(round1.get(args.mode_a, {}).get("throughput", 0.0)),
        ]
        round1_b = [
            float(round1.get(args.mode_b, {}).get("ttft", 0.0)),
            float(round1.get(args.mode_b, {}).get("latency", 0.0)),
            float(round1.get(args.mode_b, {}).get("throughput", 0.0)),
        ]
        round2_a = [sum_a["ttft_median"], sum_a["latency_median"], sum_a["throughput_median"]]
        round2_b = [sum_b["ttft_median"], sum_b["latency_median"], sum_b["throughput_median"]]

        fig, axes = plt.subplots(1, 3, figsize=(12, 4.2))
        for i, ax in enumerate(axes):
            x = [0, 1]
            w = 0.35
            ax.bar([v - w / 2 for v in x], [round1_a[i], round2_a[i]], w, label=f"{args.mode_a}")
            ax.bar([v + w / 2 for v in x], [round1_b[i], round2_b[i]], w, label=f"{args.mode_b}")
            ax.set_xticks(x)
            ax.set_xticklabels(["round1(single)", "round2(median of 5)"])
            ax.set_title(m[i])
            ax.grid(True, alpha=0.25)
        axes[0].legend(loc="best")
        fig.suptitle(f"Retest point c={args.concurrency}, tokens={args.input_tokens}")
        args.output_fig.parent.mkdir(parents=True, exist_ok=True)
        plt.tight_layout()
        plt.savefig(args.output_fig, dpi=150)
        plt.close(fig)

    print(f"Saved summary: {args.output_json}")
    print(f"Saved figure : {args.output_fig}")


if __name__ == "__main__":
    main()
