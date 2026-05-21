# FA1 vs FA2 Same-Head Observations

This note summarizes what we directly observed from the same-head FA1 vs FA2
CUDA experiment. It intentionally focuses on measured evidence, not general
FlashAttention concepts.

## Scope

FA1 v1.0.9 does not support Qwen3-8B native GQA shape (`Q heads=32`, `KV
heads=8`), so this comparison uses a same-head workload:

| Field | Value |
|---|---:|
| Q heads | `32` |
| K heads | `32` |
| V heads | `32` |
| head dim | `128` |
| dtype | BF16 |
| attention | causal |
| GPU | RTX 4090 / SM89 |

Interface alignment:

| Backend | Interface |
|---|---|
| FA1 | `flash_attn_unpadded_func` |
| FA2 | `flash_attn_varlen_func` |

Artifacts:

- `results/tables/Qwen3-8B/fa1_fa2_same_heads/fa1_latency_sweep.json`
- `results/tables/Qwen3-8B/fa1_fa2_same_heads/fa2_varlen_latency_sweep.json`
- `results/analysis/profiling/ncu/fa1_fa2_same_heads/same_heads_fa1_fa2_compare.csv`
- `results/analysis/profiling/ncu/fa1_fa2_same_heads/same_heads_fa1_fa2_compare.json`

## Main Table

| Backend | Batch | Median Latency (ms) | Est. TFLOP/s | NCU Duration (ms) | SM % | DRAM % | Est. DRAM GB | L2 Hit % | Regs | Smem KB | Occ % | Eligible Warps / Scheduler | Issue % | Waves / SM | Top Stall | Speedup |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---|---:|
| FA1 | 1 | 11.262 | 48.82 | 11.413 | 16.71 | 86.30 | 9.68 | 50.77 | 255 | 40.6 | 16.58 | 0.12 | 11.20 | 1 | L1TEX scoreboard dependency (38.4%) | 1.00x |
| FA2 | 1 | 4.061 | 135.39 | 4.267 | 44.34 | 6.06 | 0.25 | 97.71 | 182 | 48.0 | 16.40 | 0.14 | 11.58 | 16 | execution pipe wait (66.0%) | 2.77x |
| FA1 | 16 | 193.835 | 45.38 | 197.516 | 15.45 | 89.49 | 173.78 | 40.08 | 255 | 40.6 | 16.63 | 0.11 | 10.18 | 2 | L1TEX scoreboard dependency (40.9%) | 1.00x |
| FA2 | 16 | 55.379 | 158.85 | 64.279 | 47.16 | 6.80 | 4.30 | 97.71 | 182 | 48.0 | 16.64 | 0.15 | 11.64 | 256 | execution pipe wait (66.3%) | 3.50x |

## What We Saw

### 1. FA2 is materially faster on the same-head workload

Measured latency speedup:

| Batch | FA1 Median (ms) | FA2 Median (ms) | FA2 Speedup |
|---:|---:|---:|---:|
| 1 | 11.262 | 4.061 | 2.77x |
| 16 | 193.835 | 55.379 | 3.50x |

This is not a small kernel-selection artifact. The NCU kernel durations track
the same trend:

| Batch | FA1 NCU Duration (ms) | FA2 NCU Duration (ms) | FA2 Speedup |
|---:|---:|---:|---:|
| 1 | 11.413 | 4.267 | 2.67x |
| 16 | 197.516 | 64.279 | 3.07x |

### 2. FA1 is memory-pressure dominated in this workload

The clearest evidence is not just latency, but the memory profile:

| Batch | FA1 DRAM % | FA2 DRAM % | FA1 Est. DRAM GB | FA2 Est. DRAM GB |
|---:|---:|---:|---:|---:|
| 1 | 86.30 | 6.06 | 9.68 | 0.25 |
| 16 | 89.49 | 6.80 | 173.78 | 4.30 |

FA1 also has much lower L2 hit rate:

| Batch | FA1 L2 Hit % | FA2 L2 Hit % |
|---:|---:|---:|
| 1 | 50.77 | 97.71 |
| 16 | 40.08 | 97.71 |

The top NCU stall reason supports the same reading:

| Backend | Top Stall |
|---|---|
| FA1 | L1TEX scoreboard dependency |
| FA2 | execution pipe wait |

For FA1, NCU says the kernel spends cycles waiting for a scoreboard dependency
on a L1TEX operation. This is a direct memory-dependency signal.

### 3. FA2 shifts the bottleneck away from DRAM

FA2 does not simply have "higher occupancy". Occupancy is similar:

| Backend | Batch 1 Occ % | Batch 16 Occ % |
|---|---:|---:|
| FA1 | 16.58 | 16.63 |
| FA2 | 16.40 | 16.64 |

The important change is where the pressure moves:

| Backend | Batch 16 SM % | Batch 16 DRAM % |
|---|---:|---:|
| FA1 | 15.45 | 89.49 |
| FA2 | 47.16 | 6.80 |

FA2 uses the GPU more like a compute-heavy kernel, while FA1 is still exposed to
memory traffic and memory dependency stalls.

### 4. FA2 exposes much more parallel work

The `waves / SM` difference is one of the strongest observations:

| Backend | Batch 1 Waves / SM | Batch 16 Waves / SM |
|---|---:|---:|
| FA1 | 1 | 2 |
| FA2 | 16 | 256 |

This does not mean FA1 lacks tiling or SRAM reuse. FA1 is still an IO-aware
tiled attention kernel. What this shows is that FA2 organizes the work into much
more parallel units for this shape.

### 5. FA1 uses more registers in this build

| Backend | Regs / Thread | Dynamic Smem / Block |
|---|---:|---:|
| FA1 | 255 | 40.6 KB |
| FA2 | 182 | 48.0 KB |

FA1 has higher register pressure while still achieving lower SM utilization.
That combination is bad: it limits resident work while failing to convert the
available work into compute throughput.

## What We Should Not Overclaim

### FA1 also does reuse

It would be wrong to say:

```text
FA1 does not reuse, FA2 reuses.
```

Both FA1 and FA2 are IO-aware tiled attention kernels. Both avoid materializing
the full attention matrix. The measured difference is not "reuse vs no reuse".

The better statement is:

```text
FA2 organizes tiled reuse and parallel work more effectively.
```

### Warp reduction is not yet separately quantified

FA1 source clearly contains softmax reduction paths, for example:

```text
/home/xuliren/repo/flash-attention-fa1/csrc/flash_attn/src/fmha_fprop_kernel_1xN.h
```

Relevant source-level operations include:

```text
softmax.reduce_max(...)
softmax.reduce_sum_before_sync_(...)
softmax.reduce_max_after_sync_(...)
softmax.reduce_sum_after_sync_(...)
```

However, the current NCU table does not yet include a direct count of shuffle /
reduction SASS instructions. So the measured conclusion should be phrased as:

```text
FA1 is much more memory-pressure dominated and memory-stall dominated.
```

not:

```text
FA1 has quantitatively more warp-reduce instructions.
```

To make the latter claim, we need a follow-up NCU/SASS pass that collects
instruction-level counters or source-correlated stall samples.

## Summary

For this same-head BF16 causal workload on RTX 4090:

- FA2 is `2.77x-3.50x` faster by latency median.
- FA1 shows high DRAM pressure: `86-89%` DRAM utilization.
- FA2 shows low DRAM pressure: `6-7%` DRAM utilization.
- FA1 top stall is L1TEX scoreboard dependency, indicating memory-dependency
  stalls.
- FA2 top stall is execution-pipe wait, indicating the bottleneck has shifted
  away from memory.
- FA2 exposes much more parallel work: batch 16 has `256` waves/SM vs FA1's `2`.
- The concrete observation is not that FA2 "invented reuse"; it is that FA2
  makes the reused tiled attention computation much more parallel and much less
  DRAM-dependent.
