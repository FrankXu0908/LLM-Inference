# Benchmark Track

This folder contains the benchmark-focused narrative and artifacts.

## Read Order

1. `benchmark/docs/project_compass.md`
2. `benchmark/docs/round1_round2_report.md`

## Execution Scripts

Scripts are currently kept in repo-level `scripts/`:
- `scripts/benchmark_vllm.py`
- `scripts/sweep_benchmark.py`
- `scripts/sweep_compare_dp2_tp2.py`
- `scripts/plot_compare_tp2_dp2_dp2ep.py`
- `scripts/run_round1_sweep.sh`
- `scripts/run_round2_retest_point.sh`

Only completed, result-backed benchmark docs are kept in this track.

## Profiling Analysis

- Entry: `benchmark/analysis/README.md`
- Tracing workflow: `benchmark/analysis/profiling/TRACING_WORKFLOW.md`
- One-command pipeline:
  - `python benchmark/analysis/profiling/run_pipeline.py --skip-missing`
  - Note: `run_pipeline.py` is post-analysis only; traces must exist first.
