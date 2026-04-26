# Project Compass

## 1) Where We Are

Current repository status:
- We have a reproducible benchmark pipeline for vLLM OpenAI-compatible serving.
- We completed a **Round-1 full sweep** comparing `tp2`, `dp2`, `dp2_ep`.
- We completed a **Round-2 anomaly-point retest** for `c=16, input=256` with repeated runs.
- We have comparison JSON and figures that support narrative conclusions.

Primary evidence artifacts:
- `results/tables/Qwen3.5-35B-A3B-GPTQ-Int4/compare_tp2_dp2_dp2ep.json`
- `results/tables/Qwen3.5-35B-A3B-GPTQ-Int4/retest_c16_in256/retest_compare_summary.json`
- `results/figures/Qwen3.5-35B-A3B-GPTQ-Int4/compare/*.png`
- `benchmark/docs/round1_round2_report.md`

## 2) What This Project Is

This is an **inference performance investigation project**, not a generic script dump.

Core question:
- Under realistic workloads, how do `TP2`, `DP2`, and `DP2+EP` compare on TTFT, latency, and throughput?

Core methodology:
- Controlled benchmark runs
- Grid sweeps
- Outlier retesting
- Structured comparison and plotting

## 3) What This Project Is Not

- Not a production serving framework
- Not a full model-training repo
- Not a random collection of profiling logs

Any script outside the main path is auxiliary/legacy unless explicitly linked in `scripts/README.md`.

## 4) Main Execution Path (for reviewers)

1. Start vLLM service in one mode (`tp2` / `dp2` / `dp2_ep`)
2. Run full sweep:
   - `scripts/run_round1_sweep.sh <mode>`
3. After all three modes:
   - `scripts/sweep_compare_dp2_tp2.py`
   - `scripts/plot_compare_tp2_dp2_dp2ep.py`
4. Retest anomaly point:
   - `scripts/run_round2_retest_point.sh dp2`
   - `scripts/run_round2_retest_point.sh dp2_ep`
5. Summarize:
   - `benchmark/docs/round1_round2_report.md`

## 5) Where We Are Going (Next 2-3 Iterations)

### Iteration A: Reliability
- Add automatic health checks before each benchmark batch.
- Add retry-and-record for failed points (instead of hard stop or silent timeout).
- Pin service startup presets per mode to avoid routing instability.

### Iteration B: Measurement Quality
- Add repeated runs for every sweep point (`n>=3`) and report median + variance.
- Separate cold-start vs steady-state metrics explicitly.
- Add confidence intervals to comparison plots.

### Iteration C: Decision Readiness
- Produce one-page “mode recommendation matrix” by workload segment:
  - low concurrency short prompt
  - high concurrency short prompt
  - long context heavy load
- Convert conclusions into deployment guidance.

## 6) Interviewer TL;DR

- We built a closed-loop workflow: **measure -> compare -> retest anomalies -> conclude**.
- We validated that a major outlier needed retest and updated conclusions based on repeated runs.
- We have a clear next roadmap focused on reliability and decision-quality metrics.
