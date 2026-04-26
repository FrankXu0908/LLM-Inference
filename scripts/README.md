# Scripts Map

This file defines which scripts are on the **main path** and which are legacy/auxiliary.

## Main Path (active)

- `benchmark_vllm.py`
Single-run benchmark driver (OpenAI-compatible endpoint).

- `sweep_benchmark.py`
Grid sweep (`concurrency x input_tokens`) for one mode.

- `sweep_compare_dp2_tp2.py`
Compare `tp2`, `dp2`, `dp2_ep` sweep outputs.

- `plot_compare_tp2_dp2_dp2ep.py`
Generate heatmaps for Round-1 comparison.

- `run_round1_sweep.sh`
Wrapper for full sweep per mode.

- `run_round2_retest_point.sh`
Wrapper for anomaly-point repeated retest.

- `compare_retest_point.py`
Summarize Round-2 and generate Round1-vs-Round2 figure.

## Secondary (useful but not required for main benchmark narrative)

- `sweep_scenarios.py`
Scenario-oriented sweeps (dialog/RAG/ultra-long style inputs).

- `plot_results.py`
Generic plotting for one sweep JSON.

## Profiling / Deep Dive (separate track)

- `profile_*`
- `summarize_*`
- `plot_prefill_breakdown.py`

These are for deeper operator/runtime analysis and are not required for baseline TP/DP comparison reporting.

## Legacy / One-off

- `run_vllm.py`, `run_transformers.py`, `run_profile.py`, `run_server.sh`

Keep for reference unless they are explicitly integrated into the main path.
