# Benchmark Track

This folder holds benchmark narratives, project evidence, profiling workflow, and case studies.

## Primary Project

- `projects/qwen3_8b_dense/README.md`
- `projects/qwen3_8b_dense/optimization_plan.md`
- `projects/qwen3_8b_dense/request_rate_capacity_single_4090.md`
- `projects/qwen3_8b_dense/awq_marlin_dp1_standard.md`
- `projects/qwen3_8b_dense/awq_marlin_dp1_long_context.md`

This is the main line of work: Qwen3-8B dense inference optimization on dual RTX 4090 PCIe.

## Secondary Case Study

- `case_studies/qwen3_5_a3b_parallel/README.md`
- `case_studies/qwen3_5_a3b_parallel/round1_round2_report.md`

This keeps the earlier Qwen3.5-A3B `TP2 / DP2 / DP2+EP` analysis available without making it the repo's main story.

## Profiling Analysis

- Entry: `analysis/README.md`
- Tracing workflow: `analysis/profiling/TRACING_WORKFLOW.md`
- Post-analysis pipeline:
  - `python benchmark/analysis/profiling/run_pipeline.py --skip-missing`

`run_pipeline.py` analyzes existing traces; collect Nsight or Torch traces before running it.

## Common Scripts

Scripts are intentionally thin:

- `scripts/run_server.sh`
- `scripts/run_vllm_bench_request_rate.sh`
- `scripts/plot_qwen3_8b_request_rate.py`
- `scripts/profile_*`
- `scripts/summarize_*`

Benchmarking uses vLLM's native `vllm bench serve` path rather than a custom benchmark client.
