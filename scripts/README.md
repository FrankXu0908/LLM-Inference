# Scripts

This folder now keeps only thin entry points for the active project.

## Serving

- `run_server.sh`
Starts vLLM from a YAML model config.

```bash
MODEL_CONFIG=configs/qwen3_8b_dense.yaml bash scripts/run_server.sh
```

## Benchmark

- `capture_vllm_metrics.sh`
Captures `/metrics` once per second while a benchmark is running.

```bash
OUT=results/tables/Qwen3-8B/baseline_a_dp1_standard/c1_metrics.prom \
bash scripts/capture_vllm_metrics.sh
```

- `run_vllm_bench_request_rate.sh`
Runs the request-rate sweep with vLLM's native benchmark client, `vllm bench serve`.

```bash
MODEL_CONFIG=configs/qwen3_8b_dense.yaml bash scripts/run_vllm_bench_request_rate.sh
```

The benchmark path intentionally uses vLLM's native client instead of the old custom OpenAI-compatible benchmark script.

- `run_vllm_bench_concurrency.sh`
Runs burst-arrival max-concurrency sweeps with `vllm bench serve`.

```bash
RESULT_DIR=results/tables/Qwen3-8B/baseline_b_dp2_long_context \
CONCURRENCIES="1 2 4 8" RANDOM_INPUT_LEN=8192 RANDOM_OUTPUT_LEN=256 NUM_PROMPTS=128 \
SEED=42 TEMPERATURE=0 \
bash scripts/run_vllm_bench_concurrency.sh
```

- `plot_qwen3_8b_request_rate.py`
Plots the curated Qwen3-8B request-rate result used in the project writeup.

## Profiling Helpers

- `profile_prefill_once_vllm.py`
- `profile_decode_once_vllm.py`
- `profile_execution_path_vllm.py`
- `summarize_nsys_prefill.py`
- `summarize_nsys_execution_path.py`
- `summarize_tp_comm_critical_path.py`

These are for trace collection and post-processing. The structured profiling pipeline lives under `benchmark/analysis/profiling/`.
