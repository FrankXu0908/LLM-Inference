# Qwen3.5-A3B Parallel Strategy Case Study

This is a secondary case study retained for comparison and background.

## Scope

- Model: `Qwen3.5-35B-A3B-GPTQ-Int4`
- Focus: serving-level parallel strategy comparison
- Modes studied: `TP2`, `DP2`, `DP2+EP`

## Status

Completed:
- Round-1 full sweep over concurrency and input length.
- Round-2 anomaly retest for `c=16, input=256`.
- Comparison report:
  - `round1_round2_report.md`

## Role in This Repo

This case study is not the primary optimization target anymore.

It remains useful as prior evidence for:
- How to structure benchmark sweeps.
- How to retest anomalies.
- How to compare parallel strategies.
- Why PCIe communication analysis matters before assuming tensor parallelism helps.
