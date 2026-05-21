# LLM Inference Optimization Lab

This repository is a measurement-driven inference optimization project.

The primary project is now:

- `Qwen3-8B` dense model
- single-GPU and `DP=2` `NVIDIA GeForce RTX 4090` serving tracks
- vLLM serving
- optimization path from fixed baseline profiling to quantization, KV cache format, PCIe TP analysis, prefill/decode separation, and finally fusion decisions

The previous `Qwen3.5-35B-A3B-GPTQ-Int4` parallel-strategy work is kept as a secondary case study.

## Start Here

Primary project:
- `benchmark/projects/qwen3_8b_dense/README.md`
- `benchmark/projects/qwen3_8b_dense/baseline_a_dp1_standard.md`
- `benchmark/projects/qwen3_8b_dense/baseline_a_dp1_long_context.md`
- `benchmark/projects/qwen3_8b_dense/baseline_b_dp2_standard.md`
- `benchmark/projects/qwen3_8b_dense/baseline_b_dp2_long_context.md`
- `benchmark/projects/qwen3_8b_dense/awq_marlin_dp1_standard.md`
- `benchmark/projects/qwen3_8b_dense/awq_marlin_dp1_long_context.md`
- `benchmark/projects/qwen3_8b_dense/quality_baseline_a_dp1_bf16.md`
- `benchmark/projects/qwen3_8b_dense/optimization_plan.md`
- `benchmark/projects/qwen3_8b_dense/request_rate_capacity_single_4090.md`

Current kernel subproject:
- `benchmark/projects/qwen3_8b_dense/subprojects/fa1_fa2_attention_kernel/README.md`
- `benchmark/projects/qwen3_8b_dense/subprojects/fa1_fa2_attention_kernel/fa2_cuda_baseline_results.md`
- `benchmark/projects/qwen3_8b_dense/subprojects/fa1_fa2_attention_kernel/fa1_vs_fa2_same_head_observations.md`
- `benchmark/projects/qwen3_8b_dense/subprojects/fa1_fa2_attention_kernel/fa2_backend_tuning_plan.md`

Secondary case study:
- `benchmark/case_studies/qwen3_5_a3b_parallel/README.md`
- `benchmark/case_studies/qwen3_5_a3b_parallel/round1_round2_report.md`

Profiling workflow:
- `benchmark/analysis/README.md`
- `benchmark/analysis/profiling/TRACING_WORKFLOW.md`

Script map:
- `scripts/README.md`

## Current Main Roadmap

1. Baseline benchmark matrix
2. Nsight Systems + Nsight Compute / Roofline profile
3. Weight quantization A/B
4. KV cache FP8 A/B
5. Optional `TP=1` vs `TP=2` PCIe communication analysis
6. Prefill / decode disaggregation small experiment
7. Attention-kernel backend study
   - completed: CUDA FA1 vs FA2 same-head comparison
   - current: tune FA2 tile / pipeline parameters from profiling evidence
   - validation targets: latency, SM utilization, DRAM traffic, occupancy, eligible warps, and stall reasons
8. Decide whether QKV / FFN fusion is worth implementing

## Repository Layout

- `benchmark/projects/qwen3_8b_dense/`
Primary Qwen3-8B dense optimization project.

- `benchmark/case_studies/qwen3_5_a3b_parallel/`
Secondary Qwen3.5-A3B parallel strategy analysis.

- `benchmark/analysis/profiling/`
Trace classification, trace organization, and post-processing tools.

- `fx/`
Model-scoped FX graph and operator analysis. Existing Qwen3.5-A3B graphs live under `fx/models/qwen3_5_35b_a3b/`; Qwen3-8B dense has its own planned workspace under `fx/models/qwen3_8b_dense/`.

- `scripts/`
Benchmark, plotting, profiling, and summarization scripts.

- `configs/`
Model and profiler configuration.
Primary Qwen3-8B config: `configs/qwen3_8b_dense.yaml`.

- `results/`
Generated local outputs. Large generated artifacts are git-ignored by default.

## Environment

- Python 3.10+
- CUDA GPUs
- `vllm`, `torch`, `transformers`, `httpx`, `matplotlib`

```

## Notes

- Keep benchmark conclusions tied to a specific model, hardware setup, and serving configuration.
- Keep `Qwen3-8B` dense optimization results separate from `Qwen3.5-A3B` parallel-strategy results.
- Treat fusion work as a final decision, not the starting point.
- For kernel tuning, every claim should include the source diff, correctness check, latency result, and Nsight Compute evidence.
