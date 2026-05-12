# Project Compass

## Where We Are

The repository has shifted from a broad collection of serving experiments into a focused inference optimization project.

Primary track:
- `Qwen3-8B` dense model
- single-GPU and `DP=2` RTX 4090 serving tracks
- vLLM serving
- fixed-baseline optimization and profiling workflow

Completed evidence so far:
- Baseline A / A-LC for the `DP=1` serving track.
- Baseline B / B-LC for the `DP=2` serving track.
- Single-card RTX 4090 request-rate capacity sweep for Qwen3-8B.
- Qwen3.5-A3B parallel-strategy case study retained as secondary evidence.
- Trace organization and profiling post-processing tools.

## Where We Are Going

The next work should proceed in this order:

1. Keep the `DP=1` and `DP=2` baseline tracks fixed and reproducible.
2. Collect Nsight Systems traces for representative baseline points.
3. Use Nsight Compute / Roofline on selected hot kernels.
4. Run weight quantization A/B within each relevant serving track.
5. Run KV cache FP8 A/B on the long-context branches.
6. Optionally compare `TP=1` vs `TP=2` and quantify PCIe communication cost.
7. Run a small prefill/decode separation experiment.
8. Decide whether QKV / FFN fusion is worth implementing.

## What This Project Is

This is a measurement-first inference optimization study.

The core question is:
- Within a fixed serving configuration, which optimizations actually improve Qwen3-8B dense serving, and which ones only move bottlenecks around?

The project uses `DP=1` and `DP=2` as separate serving tracks. Cross-track comparisons are context, not the central claim.

## What This Project Is Not

- Not a generic benchmark dump.
- Not a production serving framework.
- Not a fusion-first kernel project.
- Not primarily about the old Qwen3.5-A3B MoE model anymore.

## Reviewer Path

Start with:
- `benchmark/projects/qwen3_8b_dense/README.md`
- `benchmark/projects/qwen3_8b_dense/optimization_plan.md`
- `benchmark/projects/qwen3_8b_dense/request_rate_capacity_single_4090.md`

Then inspect:
- `benchmark/analysis/profiling/TRACING_WORKFLOW.md`
- `benchmark/case_studies/qwen3_5_a3b_parallel/README.md`

## Interviewer TL;DR

The project now has a clear main arc:

`baseline tracks -> profile -> same-track A/B optimize -> optional PCIe analysis -> test PD split -> decide fusion`.

The older A3B work remains useful, but it is now supporting evidence rather than the headline.
