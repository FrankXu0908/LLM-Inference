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
7. Attention-kernel backend study
8. Decide whether QKV / FFN fusion is worth implementing

## Current Evidence

Completed:
- Optimization summary across BF16, AWQ-Marlin, and FP8 KV:
  - `optimization_summary.md`
  - `assets/optimization_summary_bf16_awq_fp8.png`
  - `data/optimization_summary_bf16_awq_fp8.json`
- AWQ-Marlin + FP8 KV combo performance:
  - `awq_marlin_kv_fp8_combo.md`
  - `assets/awq_marlin_kv_fp8_combo.png`
  - `data/awq_marlin_kv_fp8_combo.json`
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
- AWQ-Marlin DP=1 quality A/B:
  - `awq_marlin_dp1_quality.md`
  - `assets/awq_marlin_dp1_quality_vs_bf16.png`
  - `data/awq_marlin_dp1_quality_vs_bf16.json`
- Triton attention decode-heavy BF16 baseline:
  - `triton_attn_decode_heavy.md`
  - `assets/triton_attn_prefill_vs_decode_heavy.png`
  - `data/triton_attn_prefill_vs_decode_heavy.json`
- FP8 KV decode-heavy A/B under fixed Triton attention:
  - `kv_fp8_decode_heavy.md`
  - `assets/kv_fp8_decode_heavy_vs_bf16_triton.png`
  - `data/kv_fp8_decode_heavy_vs_bf16_triton.json`
- FP8 KV long-prefill A/B under fixed Triton attention:
  - `kv_fp8_long_prefill.md`
  - `assets/kv_fp8_long_prefill_vs_bf16_triton.png`
  - `data/kv_fp8_long_prefill_vs_bf16_triton.json`
- Triton attention integration context:
  - `triton_fa1_integration_plan.md`
  - `kernel_level_attention_analysis.md`
  - `configs/qwen3_8b_dense_triton_attn.yaml`
- FA1 to FA2 CUDA attention kernel study:
  - `subprojects/fa1_fa2_attention_kernel/README.md`
  - `subprojects/fa1_fa2_attention_kernel/experiment_matrix.md`
  - `subprojects/fa1_fa2_attention_kernel/fa2_cuda_baseline_results.md`
  - `subprojects/fa1_fa2_attention_kernel/fa1_vs_fa2_same_head_observations.md`
- FA2 backend tuning plan:
  - `subprojects/fa1_fa2_attention_kernel/fa2_backend_tuning_plan.md`
  - Baseline dispatch observed for Qwen3-8B / RTX 4090 / BF16 / causal / head_dim=128:
    `Flash_fwd_kernel_traits<128, 64, 64, 4, false, false, bf16>`
  - Current experiments change one FA2 backend variable at a time: tile shape,
    pipeline / stage control, or `BLOCK_N=128` regression check.

In progress / next:
- FA2 backend tuning from kernel-profile evidence:
  - adjust tile parameters
  - locate and adjust the real pipeline / staging control
  - test the `BLOCK_N=128` hypothesis against the current `BN64` baseline
- PD separation experiment for long-prefill prefill/decode interference.
  - `pd_routing_experiment.md`
  - `pd_routing_results.md`
  - `assets/pd_routing_mixed_vs_routed.png`
  - `data/pd_routing_mixed_vs_routed.json`
- Trace collection standardization for Nsight Systems and Nsight Compute.
- Weight quantization A/B inside each serving track:
  - Baseline B / B-LC for `DP=2`
- KV cache FP8 A/B remaining branches:
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
