# Optimization Plan

## Phase 1: Baseline Benchmark Matrix

Purpose:
- Establish the stable reference point for every later A/B test.

Baseline tracks:
- `Baseline A`: `DP=1`, short-context burst serving.
- `Baseline A-LC`: `DP=1`, long-context burst serving.
- `Baseline B`: `DP=2`, short-context burst serving.
- `Baseline B-LC`: `DP=2`, long-context burst serving.

Baseline A setup:
- Model: `Qwen3-8B` dense
- GPU: single `RTX 4090`
- Stack: `vLLM`
- Parallelism: `TP=1`, `DP=1`
- dtype: `bf16` / `fp16`
- No weight quantization
- No KV FP8
- No speculative decoding
- No prefill/decode disaggregation
- Prompt/output: `512 / 256`
- Concurrency points: `1 / 4 / 8 / 16`
- Arrival: burst arrival

Baseline B uses the same model and serving stack with `TP=1`, `DP=2`. The short-context branch keeps `512 / 256`; the long-context branch uses `8192 / 256`.

Important framing:
- The project does not optimize by claiming `DP=2` beats `DP=1`.
- `DP=1` and `DP=2` are separate serving tracks.
- Quantization, KV FP8, PD separation, and profiling conclusions should be compared against the matching baseline in the same track first.

Measure:
- Request-rate sweep.
- Concurrency sweep.
- Prompt/output length sweep.
- Prefill-heavy, decode-heavy, and balanced workloads.

Primary metrics:
- Output token throughput.
- Total token throughput.
- Mean / P99 TTFT.
- Mean / P99 TPOT or ITL.
- Failed requests and saturation behavior.

Quality guardrail:
- `quality_baseline_a_dp1_bf16.md` records the BF16 `DP=1` lm-eval baseline.
- Quantization and KV-cache experiments should report both performance movement and quality deltas against the matching quality baseline.

## Phase 2: Nsight Profile

Purpose:
- Explain the baseline before optimizing it.

Use:
- Nsight Systems for timeline, scheduling, CPU/GPU gaps, NCCL, CUDA API overhead.
- Nsight Compute / Roofline for selected hot kernels.

Questions:
- Are we compute-bound, memory-bound, launch-bound, or communication-bound?
- Does decode become small-kernel dominated?
- Does PCIe communication expose a TP critical path?

## Phase 3: Weight Quantization A/B

Purpose:
- Check whether reduced weight bandwidth improves throughput or latency.

Compare:
- `DP=1` dense baseline vs `DP=1` weight-quantized variant.
- `DP=2` dense baseline vs `DP=2` weight-quantized variant.

Keep fixed:
- Model, prompt/output lengths, request rates, max concurrency, serving flags.
- Quality task set and `lm_eval` model arguments when checking regression.

## Phase 4: KV Cache FP8 A/B

Purpose:
- Test whether KV memory pressure is a bottleneck.

Compare:
- A-LC default KV cache dtype vs A-LC FP8 KV cache.
- B-LC default KV cache dtype vs B-LC FP8 KV cache.

Watch:
- Long-context TTFT and decode ITL.
- Accuracy/quality constraints if applicable.

## Phase 5: Optional TP=1 vs TP=2 PCIe Analysis

Purpose:
- Understand whether tensor parallelism introduces PCIe/NCCL communication bottlenecks.

Compare:
- A fixed workload under `TP=1`.
- The same workload under `TP=2` over PCIe.

This is supporting analysis, not the main optimization story.

Profile:
- NCCL all-reduce/all-gather duration.
- Communication/computation overlap.
- Tail latency under load.

## Phase 6: Prefill / Decode Disaggregation

Purpose:
- Test whether separating prefill-heavy and decode-heavy execution improves tail latency.

Scope:
- Small experiment only.
- Keep it as a serving-layout experiment, not the main optimization path unless the result is strong.

## Phase 7: QKV / FFN Fusion Decision

Purpose:
- Decide whether custom fusion is worth the engineering cost.

Only proceed if:
- Profiling shows repeated small kernels or unfused QKV/FFN patterns on the critical path.
- Existing vLLM kernels do not already cover the bottleneck.
- Expected gain is larger than measurement noise and maintenance cost.
