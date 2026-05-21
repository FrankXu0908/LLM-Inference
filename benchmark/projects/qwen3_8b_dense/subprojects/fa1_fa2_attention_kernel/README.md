# FA1 to FA2 Attention Kernel Study

This is a subproject under the main Qwen3-8B dense inference optimization project.

The main project stays focused on end-to-end vLLM inference optimization. This
subproject focuses only on attention-kernel behavior.

## Question

Explain why FlashAttention-2 CUDA kernels are faster than FlashAttention-1 CUDA
kernels, and when they are not faster.

The purpose is not to claim that one implementation is universally better. The
purpose is to isolate which kernel-level factors matter for Qwen3-8B on RTX
4090.

## Fixed Target

- Model: `Qwen3-8B`
- GPU: RTX 4090 / Ada / SM89
- dtype: BF16
- attention type: causal only
- dropout: false
- alibi: false
- local attention: false
- softcap: false
- primary kernel source: `/home/xuliren/repo/flash-attention`
- serving stack for later integration checks: vLLM

Model-derived shape constants:

| Field | Value |
|---|---:|
| hidden size | `4096` |
| layers | `36` |
| query heads | `32` |
| KV heads | `8` |
| GQA ratio | `4` |
| head dim | `128` |
| intermediate size | `12288` |
| max position embeddings | `40960` |

## Variables Under Study

Only change attention-kernel implementation details:

- FA1 CUDA vs FA2 CUDA kernel structure
- tile size
- block scheduling
- warp partitioning
- shared-memory layout
- pipeline / staging strategy

Do not mix in:

- AWQ
- FP8 KV
- speculative decoding
- PD routing
- tensor parallelism
- serving scheduler changes

Those belong to the main inference optimization project, not this kernel study.

## Workloads

Primary benchmark shape:

| Batch | Seq Len | Q Heads | KV Heads | Head Dim | Causal |
|---:|---:|---:|---:|---:|---|
| `1` | `8192` | `32` | `8` | `128` | yes |
| `4` | `8192` | `32` | `8` | `128` | yes |
| `16` | `8192` | `32` | `8` | `128` | yes |

Additional shape sweep:

- sequence length: `512`, `2048`, `8192`, `16384`
- batch: `1`, `4`, `16`
- dtype: BF16

## Measurement Stack

Use two levels of measurement.

### 1. Standalone CUDA Microbenchmark

Purpose:

- Compare FA1 vs FA2 CUDA kernels without vLLM scheduler, paged KV, HTTP
  serving, or request batching.

Metrics:

- latency
- TFLOP/s
- effective memory bandwidth
- numerical error vs reference
- max allocated memory

### 2. vLLM Integrated Kernel Profile

Purpose:

- Verify whether the microbenchmark result survives vLLM integration.
- Detect extra layout conversion, KV cache, or metadata kernels.

Metrics:

- Nsight Systems kernel duration
- Nsight Compute kernel duration
- SM / Tensor Core utilization
- shared memory behavior
- occupancy
- stall reasons

## Current FA2 Dispatch Hypothesis

For Qwen3-8B on RTX 4090:

```text
dtype = BF16
causal = true
dropout = false
head_dim = 128
arch = SM89
```

the FlashAttention source dispatches:

```text
Flash_fwd_kernel_traits<128, 64, 64, 4, false, false, bf16>
```

So the first standalone FA2 CUDA experiment should verify `BM64 BN64 warps4`,
not the earlier rough assumption of `BM128 BN64`.

Details:

- `benchmark/projects/qwen3_8b_dense/subprojects/fa1_fa2_attention_kernel/fa2_cuda_dispatch_notes.md`
- `benchmark/projects/qwen3_8b_dense/subprojects/fa1_fa2_attention_kernel/fa2_cuda_baseline_results.md`
- `benchmark/projects/qwen3_8b_dense/subprojects/fa1_fa2_attention_kernel/fa1_vs_fa2_same_head_observations.md`
- `benchmark/projects/qwen3_8b_dense/subprojects/fa1_fa2_attention_kernel/fa2_backend_tuning_plan.md`

## Current FA2 Backend Tuning Track

The next step is no longer just observing FA2. It is to modify the FA2 backend
and validate concrete hypotheses:

- adjust tile parameters from the current `BM64 BN64 warps4` baseline
- locate and adjust the real pipeline / staging control in source
- test whether `BLOCK_N = 128` helps or regresses this RTX 4090 workload

Each variant must be evaluated with:

- correctness vs baseline FA2
- standalone latency
- Nsight Compute duration, SM %, DRAM %, registers, shared memory, occupancy,
  eligible warps, waves per SM, and top stall reason

## Related vLLM Triton Baseline

This is not the main FA1/FA2 CUDA baseline. It is a related integrated-attention
profile from the previous vLLM `TRITON_ATTN` work, useful as context when we
later compare standalone kernels against vLLM integration.

Baseline: vLLM `TRITON_ATTN` on original BF16 weights.

Target kernel:

```text
kernel_unified_attention_2d
```

Workload:

```text
input tokens = 8192
output tokens = 1
batch / concurrency = 1
dtype = BF16
attention = causal
```

Artifacts:

- `results/traces/ncu/prefill/qwen3_8b_dp1_bf16_triton/baseline_prefill_8192_unified_attention_run*/`
- `results/analysis/profiling/ncu/qwen3_8b_triton_prefill_8192_unified_attention_runs/ncu_runs_summary.csv`
- `results/analysis/profiling/ncu/qwen3_8b_triton_prefill_8192_unified_attention_runs/ncu_runs_summary.json`

Median NCU baseline over 3 runs:

| Metric | Median |
|---|---:|
| kernel duration | `8.68 ms` |
| grid size | `16392` |
| block size | `128` |
| waves / SM | `32.02` |
| compute throughput | `59.49%` |
| memory throughput | `16.39 GB/s` |
| DRAM throughput | `1.67%` |
| L2 hit rate | `99.71%` |
| L1/TEX hit rate | `5.89%` |
| SM Busy | `30.20%` |
| Mem Busy | `87.84%` |
| issue slots busy | `30.20%` |
| no eligible warp | `69.81%` |
| eligible warps / scheduler | `0.38` |
| registers / thread | `91` |
| dynamic shared memory / block | `21.76 KB` |
| theoretical occupancy | `33.33%` |
| achieved occupancy | `32.89%` |
| branch efficiency | `40.03%` |

Current interpretation:

- The baseline is stable across repeated NCU runs.
- It is not DRAM-bandwidth limited.
- It has very high L2 hit rate.
- The interesting bottlenecks are scheduler eligibility, shared-memory access
  behavior, and branch/divergence behavior.

## Hypotheses

FA2-style kernels may win by:

- improving work partitioning across warps and blocks
- reducing non-matmul overhead around online softmax
- improving shared-memory reuse / access layout
- improving eligible warp availability
- reducing synchronization and dependency stalls

FA2 may not win when:

- batch/sequence shape is too small to fill the GPU
- vLLM integration requires extra layout conversion
- shared-memory/register pressure lowers occupancy too much
- paged KV metadata dominates the kernel path
- the workload is decode-heavy rather than prefill-heavy

## Success Criteria

For the same shape and dtype, a candidate FA kernel is better only if it improves
at least one of the following without regressing correctness:

- lower median kernel duration
- higher SM / Tensor Core utilization
- lower `No Eligible`
- higher eligible warps per scheduler
- fewer shared-memory conflict warnings
- lower total integrated attention time in Nsight Systems

Serving-level speedup is a separate claim and requires a separate vLLM A/B.
