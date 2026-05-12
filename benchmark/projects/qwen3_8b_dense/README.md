# Qwen3-8B Dense Inference Optimization

This is the primary project in this repository.

## Target

- Model: `Qwen3-8B` dense
- Hardware target: `NVIDIA GeForce RTX 4090` single-GPU and DP=2 serving tracks
- Serving stack: `vLLM`
- Main goal: build a measurement-driven optimization path from baseline profiling to selected framework/kernel-level improvements.
- Config: `configs/qwen3_8b_dense.yaml`

## Project Order

1. Baseline benchmark matrix
2. Nsight Systems + Nsight Compute / Roofline profile
3. Weight quantization A/B
4. KV cache FP8 A/B
5. Optional `TP=1` vs `TP=2` PCIe communication analysis
6. Prefill / decode disaggregation small experiment
7. Decide whether QKV / FFN fusion is worth implementing

## Current Evidence

Completed:
- Baseline A: single GPU serving baseline:
  - `baseline_a_dp1_standard.md`
  - `assets/baseline_a_dp1_standard_concurrency.png`
- Baseline A-LC: single GPU long-context branch:
  - `baseline_a_dp1_long_context.md`
  - `assets/baseline_a_dp1_long_context_concurrency.png`
- Baseline B: DP=2 standard baseline:
  - `baseline_b_dp2_standard.md`
  - `assets/baseline_b_dp2_standard_concurrency.png`
- Baseline B-LC: DP=2 long-context branch:
  - `baseline_b_dp2_long_context.md`
  - `assets/baseline_b_dp2_long_context_concurrency.png`
- Single-card RTX 4090 request-rate capacity sweep:
  - `request_rate_capacity_single_4090.md`
  - `data/request_rate_sweep_512in_256out_single_4090.json`
  - `assets/request_rate_capacity_single_4090.png`
- Quality Baseline A: DP=1 BF16 guardrail:
  - `quality_baseline_a_dp1_bf16.md`
  - `results/eval/qwen3_8b/baseline_a_dp1_bf16/results_2026-05-11T23-20-17.119348.json`
- AWQ-Marlin DP=1 standard A/B:
  - `awq_marlin_dp1_standard.md`
  - `assets/awq_marlin_dp1_standard_vs_baseline_a.png`
  - `results/tables/Qwen3-8B/awq_marlin_dp1_standard/awq_marlin_dp1_standard_vs_baseline_a_summary.json`
- AWQ-Marlin DP=1 long-context A/B:
  - `awq_marlin_dp1_long_context.md`
  - `assets/awq_marlin_dp1_long_context_vs_baseline_a_lc.png`
  - `results/tables/Qwen3-8B/awq_marlin_dp1_long_context/awq_marlin_dp1_long_context_vs_baseline_a_lc_summary.json`

In progress / next:
- Trace collection standardization for Nsight Systems and Nsight Compute.
- Weight quantization A/B inside each serving track:
  - Baseline B / B-LC for `DP=2`
- KV cache FP8 A/B primarily on long-context branches:
  - A-LC for single-GPU behavior
  - B-LC for DP=2 serving behavior

## Baseline Tracks

The project is not centered on proving that `DP=2` is faster than `DP=1`.

Instead, `DP=1` and `DP=2` are two serving configurations with their own baselines. Optimizations are evaluated within the same configuration first:

- `DP=1`: compare Baseline A / A-LC against DP=1 quantization, KV FP8, or profiling changes.
- `DP=2`: compare Baseline B / B-LC against DP=2 quantization, KV FP8, PD separation, or profiling changes.

Cross-track comparisons are useful context, but they are not the main optimization claim.

Start server with the project config:

```bash
MODEL_CONFIG=configs/qwen3_8b_dense.yaml bash scripts/run_server.sh
```

Run the request-rate baseline sweep with vLLM's native benchmark client:

```bash
MODEL_CONFIG=configs/qwen3_8b_dense.yaml bash scripts/run_vllm_bench_request_rate.sh
```

The underlying command is `vllm bench serve` with random prompts, fixed input/output length, and a swept `--request-rate`.

## Decision Rule

Each optimization should have:
- A fixed baseline configuration.
- One changed variable.
- Throughput, TTFT, TPOT/ITL, and tail-latency metrics.
- Profiling evidence showing why the result changed.

Fusion work is deliberately last. We only pursue QKV / FFN fusion if profiling shows a meaningful kernel-launch or memory-traffic bottleneck that is not already solved by vLLM, quantization, KV cache format, or parallelism choice.
