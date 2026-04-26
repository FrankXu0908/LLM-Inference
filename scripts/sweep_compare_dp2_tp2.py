import argparse
import json
from pathlib import Path
from model_config import load_model_name


def pct_delta(new: float, old: float) -> float | None:
    if not old:
        return None
    return (new - old) / old * 100.0


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Compare TP2, DP2 and DP2+EP sweep result files")
    parser.add_argument("--model", type=str, default=None)
    parser.add_argument("--model-config", type=str, default="configs/model.yaml")
    parser.add_argument("--results-root", type=Path, default=Path("results/tables"))
    parser.add_argument("--tp2-results", type=Path, default=None)
    parser.add_argument("--dp2-results", type=Path, default=None)
    parser.add_argument("--dp2-ep-results", type=Path, default=None)
    parser.add_argument("--output", type=Path, default=None)
    return parser.parse_args()


def load_rows(path: Path) -> list[dict]:
    with open(path, "r", encoding="utf-8") as f:
        data = json.load(f)
    if not isinstance(data, list):
        raise ValueError(f"Expected list JSON in {path}, got {type(data)}")
    return data


def main() -> None:
    args = parse_args()
    if args.model is None:
        args.model = load_model_name(args.model_config)
    out_dir = args.results_root / Path(args.model).name
    if args.tp2_results is None:
        args.tp2_results = out_dir / "tp2" / "results.json"
    if args.dp2_results is None:
        args.dp2_results = out_dir / "dp2" / "results.json"
    if args.dp2_ep_results is None:
        args.dp2_ep_results = out_dir / "dp2_ep" / "results.json"
    if args.output is None:
        args.output = out_dir / "compare_tp2_dp2_dp2ep.json"

    for p in [args.tp2_results, args.dp2_results, args.dp2_ep_results]:
        if not p.exists():
            raise FileNotFoundError(f"Results file not found: {p}")

    tp2_rows = load_rows(args.tp2_results)
    dp2_rows = load_rows(args.dp2_results)
    dp2_ep_rows = load_rows(args.dp2_ep_results)

    tp2_map = {(x.get("concurrency"), x.get("input_tokens")): x for x in tp2_rows if isinstance(x, dict)}
    dp2_map = {(x.get("concurrency"), x.get("input_tokens")): x for x in dp2_rows if isinstance(x, dict)}
    dp2_ep_map = {(x.get("concurrency"), x.get("input_tokens")): x for x in dp2_ep_rows if isinstance(x, dict)}

    keys = sorted(set(tp2_map.keys()) | set(dp2_map.keys()) | set(dp2_ep_map.keys()))
    compare = []
    for key in keys:
        c, t = key
        tp2 = tp2_map.get(key)
        dp2 = dp2_map.get(key)
        dp2_ep = dp2_ep_map.get(key)

        item = {
            "concurrency": c,
            "input_tokens": t,
            "tp2": tp2,
            "dp2": dp2,
            "dp2_ep": dp2_ep,
            "delta_dp2_vs_tp2": None,
            "delta_dp2ep_vs_tp2": None,
            "delta_dp2ep_vs_dp2": None,
        }

        if tp2 and dp2:
            item["delta_dp2_vs_tp2"] = {
                "ttft_pct": pct_delta(float(dp2.get("ttft", 0.0)), float(tp2.get("ttft", 0.0))),
                "latency_pct": pct_delta(float(dp2.get("latency", 0.0)), float(tp2.get("latency", 0.0))),
                "throughput_pct": pct_delta(
                    float(dp2.get("throughput", 0.0)),
                    float(tp2.get("throughput", 0.0)),
                ),
            }
        if tp2 and dp2_ep:
            item["delta_dp2ep_vs_tp2"] = {
                "ttft_pct": pct_delta(float(dp2_ep.get("ttft", 0.0)), float(tp2.get("ttft", 0.0))),
                "latency_pct": pct_delta(float(dp2_ep.get("latency", 0.0)), float(tp2.get("latency", 0.0))),
                "throughput_pct": pct_delta(
                    float(dp2_ep.get("throughput", 0.0)),
                    float(tp2.get("throughput", 0.0)),
                ),
            }
        if dp2 and dp2_ep:
            item["delta_dp2ep_vs_dp2"] = {
                "ttft_pct": pct_delta(float(dp2_ep.get("ttft", 0.0)), float(dp2.get("ttft", 0.0))),
                "latency_pct": pct_delta(float(dp2_ep.get("latency", 0.0)), float(dp2.get("latency", 0.0))),
                "throughput_pct": pct_delta(
                    float(dp2_ep.get("throughput", 0.0)),
                    float(dp2.get("throughput", 0.0)),
                ),
            }

        compare.append(item)

    args.output.parent.mkdir(parents=True, exist_ok=True)
    with open(args.output, "w", encoding="utf-8") as f:
        json.dump(compare, f, indent=2, ensure_ascii=False)
    print(f"saved {args.output}")


if __name__ == "__main__":
    main()
