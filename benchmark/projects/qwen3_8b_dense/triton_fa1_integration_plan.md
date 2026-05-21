# Triton FA1 Integration Plan

## Goal

Integrate an external FlashAttention-1-style Triton implementation into vLLM and compare it against vLLM's existing Triton attention implementation.

This is a kernel/framework-level study, not a FlashAttention extension benchmark.

We are not comparing against vLLM's `FLASH_ATTN` backend. The target is:

- Baseline: vLLM current `TRITON_ATTN` path.
- Experiment: external FA1-style Triton kernel wired into vLLM as an alternate Triton attention path.

## Why This Path

The serving-level optimizations so far showed that:

- Weight quantization improves decode-heavy throughput by reducing weight bandwidth pressure.
- FP8 KV improves KV residency and decode-heavy behavior.
- Long-prefill still creates attention/prefill pressure and can pollute decode latency.

That makes attention-kernel behavior worth studying, especially under:

- long prefill: `8192 in / 256 out`
- decode-heavy: `256 in / 8192 out`
- standard serving: `512 in / 256 out`

## vLLM Integration Target

Start from the existing Triton attention files in vLLM:

- `vllm/v1/attention/backends/triton_attn.py`
- `vllm/v1/attention/ops/triton_prefill_attention.py`
- `vllm/v1/attention/ops/triton_decode_attention.py`
- `vllm/v1/attention/ops/triton_unified_attention.py`

The cleanest first integration is not to replace everything.

Instead:

1. Add a new FA1-style Triton prefill op file.
2. Add a backend switch or config flag that selects this op for prefill.
3. Keep decode unchanged at first.
4. Compare only the prefill-heavy impact first.
5. Extend to decode only if the external implementation has a decode kernel too.

## Recommended Integration Shape

Add a new op module in the vLLM repo:

```text
vllm/v1/attention/ops/triton_fa1_prefill_attention.py
```

Then wire it behind an explicit selector, for example:

```text
TRITON_ATTN_IMPL=fa1_prefill
```

or a vLLM config flag if we want this to be more formal.

For early experiments, an environment variable is simpler and less invasive.

## Correctness First

Before benchmarking serving:

1. Run a standalone correctness test against PyTorch SDPA or vLLM's current Triton output.
2. Test the exact Qwen3-8B shapes.
3. Test BF16 first.
4. Use small tolerances appropriate for attention numerics.

Minimum shape set:

```text
batch = 1
num_q_heads = Qwen3-8B attention heads
num_kv_heads = Qwen3-8B KV heads
head_dim = Qwen3-8B head dim
seq_len = 256, 2048, 8192
dtype = bf16
```

Do not start with paged KV, FP8 KV, AWQ, or serving concurrency. Those make debugging much harder.

## Benchmark Stages

### Stage 0: Baseline Triton Kernel Profile

Before integrating any external FA1 kernel, capture the current original-weight
vLLM `TRITON_ATTN` baseline.

Prefill-oriented trace:

```bash
MODEL_CONFIG=configs/qwen3_8b_dense_triton_attn.yaml \
PHASE=prefill \
INPUT_TOKENS=8192 \
OUTPUT_TOKENS=1 \
RUN_ID=baseline_prefill_8192 \
bash scripts/run_qwen3_8b_triton_nsys_profile.sh
```

Decode-oriented trace:

```bash
MODEL_CONFIG=configs/qwen3_8b_dense_triton_attn.yaml \
PHASE=decode \
INPUT_TOKENS=256 \
OUTPUT_TOKENS=8192 \
RUN_ID=baseline_decode_256_8192 \
bash scripts/run_qwen3_8b_triton_nsys_profile.sh
```

Generated traces follow the standard layout:

```text
results/traces/nsys/<phase>/qwen3_8b_dp1_bf16_triton/<run_id>/
```

Use this trace to identify the current Triton attention kernel names, durations,
and surrounding layout/cache kernels before changing code.

For kernel-level profiling, continue with:

- `kernel_level_attention_analysis.md`
- `scripts/run_qwen3_8b_triton_ncu_profile.sh`

### Stage 1: Kernel Microbenchmark

Purpose: isolate FA1-style Triton kernel behavior.

Measure:

- kernel latency
- achieved bandwidth
- achieved TFLOP/s if applicable
- max memory allocated
- numerical error vs reference

Shapes:

- `seq_len=256`
- `seq_len=2048`
- `seq_len=4096`
- `seq_len=8192`

This answers whether the FA1 kernel is competitive before vLLM integration overhead enters the picture.

### Stage 2: vLLM Single-Prefill Test

Purpose: verify the kernel works inside vLLM execution.

Workload:

```text
input=8192
output=1
concurrency=1
num_prompts=16
```

Metrics:

- TTFT
- CUDA kernel time from Nsight Systems
- attention kernel name and duration
- any extra layout/transpose kernels introduced by integration

### Stage 3: Serving A/B

Run the same serving matrices as the rest of the project.

Standard:

```text
input=512
output=256
concurrency=1/4/8/16
```

Long-prefill:

```text
input=8192
output=256
concurrency=1/2/4/8
```

Decode-heavy:

```text
input=256
output=8192
concurrency=1/2/4/8
```

Expected: if only prefill is replaced, the strongest signal should appear in TTFT and long-prefill throughput. Decode-heavy ITL may not improve much unless decode attention is also changed.

## What Would Count As Success

A useful FA1 integration does not need to beat vLLM everywhere.

Useful outcomes:

- Faster long-prefill TTFT at low concurrency.
- Lower prefill attention CUDA time in Nsight Systems.
- Fewer or shorter attention kernels.
- Clear explanation of why it loses, if it loses.

Not useful:

- A benchmark speedup caused by changing scheduler settings.
- A speedup mixed with AWQ or FP8 KV.
- A serving result where correctness has not been checked.

## Risks

Main risk: external FA1 kernels often assume contiguous Q/K/V layout, while vLLM serving uses paged KV and metadata-heavy attention paths.

If the integration requires extra reshape, gather, scatter, or cache conversion kernels, those can erase the attention-kernel gain.

This is why the first comparison must include Nsight Systems kernel timelines, not only throughput.

## Decision Rule

Proceed to deeper integration only if Stage 1 and Stage 2 show a real prefill attention improvement.

If the external FA1 Triton kernel is slower or needs expensive layout conversion, document it and stop. That is still a valid optimization result: it shows vLLM's existing Triton path is already better aligned with serving constraints.
