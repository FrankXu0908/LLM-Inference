import argparse
import json
import subprocess
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List
from model_config import load_model_name


@dataclass
class SweepConfig:
    scenario: str
    prompt_type: str
    input_tokens: int
    max_tokens: int
    num_requests: int
    concurrency_values: List[int]


def default_scenarios() -> List[SweepConfig]:
    return [
        SweepConfig(
            scenario="dialog_qa",
            prompt_type="short",
            input_tokens=256,
            max_tokens=128,
            num_requests=64,
            concurrency_values=[1, 2, 4, 8, 16, 32],
        ),
        SweepConfig(
            scenario="rag_longdoc",
            prompt_type="rag",
            input_tokens=4096,
            max_tokens=256,
            num_requests=32,
            concurrency_values=[1, 2, 4, 8, 12, 16],
        ),
        SweepConfig(
            scenario="ultra_long_1to4",
            prompt_type="long",
            input_tokens=20000,
            max_tokens=64,
            num_requests=8,
            concurrency_values=[1, 2, 3, 4],
        ),
    ]


def run_one(
    model: str,
    base_url: str,
    api_key: str,
    tokenizer: str,
    cfg: SweepConfig,
    concurrency: int,
    output_json: Path,
    timeout: float,
    no_stream: bool,
    dry_run: bool,
) -> Dict:
    cmd = [
        "python",
        "scripts/benchmark_vllm.py",
        "--model",
        model,
        "--base-url",
        base_url,
        "--api-key",
        api_key,
        "--prompt-type",
        cfg.prompt_type,
        "--input-tokens",
        str(cfg.input_tokens),
        "--max-tokens",
        str(cfg.max_tokens),
        "--num-requests",
        str(cfg.num_requests),
        "--concurrency",
        str(concurrency),
        "--timeout",
        str(timeout),
        "--output-json",
        str(output_json),
    ]
    if tokenizer:
        cmd += ["--tokenizer", tokenizer]
    if no_stream:
        cmd += ["--no-stream"]

    print(f"[run] scenario={cfg.scenario} conc={concurrency} tokens={cfg.input_tokens}")
    print(" ".join(cmd))
    if dry_run:
        return {
            "scenario": cfg.scenario,
            "concurrency": concurrency,
            "input_tokens": cfg.input_tokens,
            "summary": {},
            "detail_path": str(output_json),
            "dry_run": True,
        }

    proc = subprocess.run(cmd, text=True, capture_output=True)
    if proc.returncode != 0:
        return {
            "scenario": cfg.scenario,
            "concurrency": concurrency,
            "input_tokens": cfg.input_tokens,
            "error": proc.stderr.strip() or proc.stdout.strip(),
            "detail_path": str(output_json),
        }

    with open(output_json, "r", encoding="utf-8") as f:
        payload = json.load(f)
    return {
        "scenario": cfg.scenario,
        "concurrency": concurrency,
        "input_tokens": cfg.input_tokens,
        "summary": payload.get("summary", {}),
        "detail_path": str(output_json),
    }


def find_degradation_point(
    rows: List[Dict],
    latency_factor: float = 1.5,
    ttft_factor: float = 1.5,
    throughput_drop: float = 0.7,
) -> Dict:
    valid = [r for r in rows if r.get("summary")]
    if not valid:
        return {"status": "no_valid_data"}

    valid = sorted(valid, key=lambda x: x["concurrency"])
    base = valid[0]["summary"]
    base_latency = max(1e-9, float(base.get("latency_mean", 0.0)))
    base_ttft = max(1e-9, float(base.get("ttft_mean", 0.0)))
    base_tps = max(1e-9, float(base.get("aggregate_output_tps", 0.0)))

    for r in valid[1:]:
        s = r["summary"]
        if float(s.get("latency_mean", 0.0)) >= base_latency * latency_factor:
            return {"status": "degraded", "metric": "latency_mean", "at_concurrency": r["concurrency"]}
        if float(s.get("ttft_mean", 0.0)) >= base_ttft * ttft_factor:
            return {"status": "degraded", "metric": "ttft_mean", "at_concurrency": r["concurrency"]}
        if float(s.get("aggregate_output_tps", 0.0)) <= base_tps * throughput_drop:
            return {"status": "degraded", "metric": "aggregate_output_tps", "at_concurrency": r["concurrency"]}

    return {"status": "stable_in_test_range"}


def plot_scenario(rows: List[Dict], output_png: Path, scenario: str) -> None:
    try:
        import matplotlib.pyplot as plt
    except Exception:
        print(f"[warn] matplotlib not available, skip figure for {scenario}")
        return

    valid = [r for r in rows if r.get("summary")]
    if not valid:
        return

    valid = sorted(valid, key=lambda x: x["concurrency"])
    xs = [r["concurrency"] for r in valid]
    latency = [r["summary"]["latency_mean"] for r in valid]
    ttft = [r["summary"]["ttft_mean"] for r in valid]
    tps = [r["summary"]["aggregate_output_tps"] for r in valid]

    fig, ax1 = plt.subplots(figsize=(9, 5))
    ax1.plot(xs, latency, marker="o", label="latency_mean (s)", color="#1f77b4")
    ax1.plot(xs, ttft, marker="s", label="ttft_mean (s)", color="#ff7f0e")
    ax1.set_xlabel("Concurrency")
    ax1.set_ylabel("Seconds")
    ax1.grid(True, alpha=0.25)

    ax2 = ax1.twinx()
    ax2.plot(xs, tps, marker="^", label="aggregate_output_tps", color="#2ca02c")
    ax2.set_ylabel("Tokens/s")

    lines_1, labels_1 = ax1.get_legend_handles_labels()
    lines_2, labels_2 = ax2.get_legend_handles_labels()
    ax1.legend(lines_1 + lines_2, labels_1 + labels_2, loc="best")
    ax1.set_title(f"{scenario}: performance vs concurrency")
    output_png.parent.mkdir(parents=True, exist_ok=True)
    plt.tight_layout()
    plt.savefig(output_png, dpi=150)
    plt.close(fig)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Scenario-based vLLM sweep and degradation detection")
    parser.add_argument("--model", type=str, default=None)
    parser.add_argument("--model-config", type=str, default="configs/model.yaml")
    parser.add_argument("--base-url", type=str, default="http://127.0.0.1:8000/v1")
    parser.add_argument("--api-key", type=str, default="EMPTY")
    parser.add_argument("--tokenizer", type=str, default="")
    parser.add_argument("--timeout", type=float, default=300.0)
    parser.add_argument("--no-stream", action="store_true")
    parser.add_argument("--dry-run", action="store_true")
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path("results/tables/Qwen3.5-35B-A3B-GPTQ-Int4/scenarios"),
    )
    parser.add_argument(
        "--fig-dir",
        type=Path,
        default=Path("results/figures/Qwen3.5-35B-A3B-GPTQ-Int4/scenarios"),
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    if args.model is None:
        args.model = load_model_name(args.model_config)
    args.output_dir.mkdir(parents=True, exist_ok=True)
    args.fig_dir.mkdir(parents=True, exist_ok=True)

    all_rows: List[Dict] = []
    scenario_summary: Dict[str, Dict] = {}
    scenario_rows: Dict[str, List[Dict]] = {}

    for cfg in default_scenarios():
        rows: List[Dict] = []
        for conc in cfg.concurrency_values:
            detail_json = args.output_dir / f"{cfg.scenario}_c{conc}.json"
            row = run_one(
                model=args.model,
                base_url=args.base_url,
                api_key=args.api_key,
                tokenizer=args.tokenizer,
                cfg=cfg,
                concurrency=conc,
                output_json=detail_json,
                timeout=args.timeout,
                no_stream=args.no_stream,
                dry_run=args.dry_run,
            )
            rows.append(row)
            all_rows.append(row)

        deg = find_degradation_point(rows)
        scenario_summary[cfg.scenario] = {
            "config": {
                "prompt_type": cfg.prompt_type,
                "input_tokens": cfg.input_tokens,
                "max_tokens": cfg.max_tokens,
                "num_requests": cfg.num_requests,
                "concurrency_values": cfg.concurrency_values,
            },
            "degradation": deg,
        }
        scenario_rows[cfg.scenario] = rows
        plot_scenario(rows, args.fig_dir / f"{cfg.scenario}.png", cfg.scenario)

    out = {
        "model": args.model,
        "base_url": args.base_url,
        "scenarios": scenario_summary,
        "rows": all_rows,
    }
    summary_path = args.output_dir / "scenario_sweep_summary.json"
    with open(summary_path, "w", encoding="utf-8") as f:
        json.dump(out, f, indent=2, ensure_ascii=False)

    print(f"Saved summary: {summary_path}")
    for name, item in scenario_summary.items():
        print(f"{name}: {json.dumps(item['degradation'], ensure_ascii=False)}")


if __name__ == "__main__":
    main()
