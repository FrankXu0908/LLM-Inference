# LLM Inference Analysis (Qwen3.5-35B-A3B-GPTQ-Int4)

This repository contains benchmarking, profiling, and comparison scripts for vLLM inference experiments on Qwen3.5-35B-A3B-GPTQ-Int4.

## Environment

- Python 3.10+
- CUDA GPUs
- `vllm`, `torch`, `transformers`, `httpx`, `matplotlib`

Recommended:
```bash
conda activate vllm
```

## Project Focus

This repo has one core narrative:
1. Run controlled benchmarks for `tp2 / dp2 / dp2_ep`.
2. Compare across TTFT / latency / throughput.
3. Retest anomalies with repeated runs.
4. Produce decision-oriented conclusions.

If you are reviewing this repo for the first time, start from:
- `benchmark/docs/project_compass.md`
- `benchmark/docs/round1_round2_report.md`
- `scripts/README.md`

Track entry points:
- Benchmark track: `benchmark/README.md`
- FX track: `fx/README.md`

## Main Scripts

- `scripts/benchmark_vllm.py`
Single benchmark run against OpenAI-compatible vLLM API.

- `scripts/sweep_benchmark.py`
Grid sweep across `(concurrency, input_tokens)`.

- `scripts/sweep_compare_dp2_tp2.py`
Compare sweep outputs of `tp2`, `dp2`, and `dp2_ep`.

- `scripts/plot_compare_tp2_dp2_dp2ep.py`
Generate heatmaps from comparison JSON.

## Typical Workflow

1. Start service in one mode (example: TP2 / DP2 / DP2+EP).
```bash
bash scripts/run_server.sh
```
2. Run sweep for that mode:
```bash
python scripts/sweep_benchmark.py --parallel-mode tp2
python scripts/sweep_benchmark.py --parallel-mode dp2
python scripts/sweep_benchmark.py --parallel-mode dp2_ep
```
3. Compare:
```bash
python scripts/sweep_compare_dp2_tp2.py
```
4. Plot:
```bash
python scripts/plot_compare_tp2_dp2_dp2ep.py
```

5. Anomaly retest (Round-2):
```bash
bash scripts/run_round2_retest_point.sh dp2
bash scripts/run_round2_retest_point.sh dp2_ep
```

## Output Layout

- `results/tables/<model>/<mode>/results.json`
- `results/tables/<model>/compare_tp2_dp2_dp2ep.json`
- `results/figures/<model>/compare/*.png`

## Notes

- `benchmark_vllm.py` supports `--parallel-mode {tp2,dp2,dp2_ep,custom}` and auto-organizes output paths.
- Large generated artifacts are git-ignored; placeholder `.gitkeep` files keep directory structure.
