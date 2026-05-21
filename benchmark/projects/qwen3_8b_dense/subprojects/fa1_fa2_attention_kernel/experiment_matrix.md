# Experiment Matrix

## Fixed Variables

| Variable | Value |
|---|---|
| model | `Qwen3-8B` |
| GPU | RTX 4090 / Ada SM89 |
| dtype | BF16 |
| attention | causal |
| dropout | false |
| alibi | false |
| local attention | false |
| softcap | false |
| query heads | `32` |
| KV heads | `8` |
| head dim | `128` |

## Independent Variables

Only these should change:

- implementation: FA1 CUDA / FA2 CUDA / vLLM Triton reference
- tile shape
- num warps
- num stages
- shared-memory layout
- pipeline strategy

## Shape Matrix

| Case | Batch | Seq Len | Purpose |
|---|---:|---:|---|
| small prefill | `1` | `512` | launch overhead / small-shape behavior |
| medium prefill | `1` | `2048` | transition behavior |
| long prefill | `1` | `8192` | primary baseline |
| very long prefill | `1` | `16384` | long-context scaling |
| batched long prefill | `4` | `8192` | occupancy / wave scaling |
| high-batch long prefill | `16` | `8192` | saturation / scheduling |

Full standalone FA2 CUDA sweep:

| Batch | Seq Len |
|---:|---:|
| `1 / 4 / 16` | `512 / 2048 / 8192 / 16384` |

## Metrics

Microbenchmark:

- median latency
- p90 latency
- TFLOP/s
- effective bandwidth
- numerical error

Nsight Compute:

- duration
- compute throughput
- tensor utilization
- memory throughput
- DRAM throughput
- L2 hit rate
- shared-memory conflicts
- achieved occupancy
- eligible warps per scheduler
- no eligible warp percentage
- top stall reasons

Nsight Systems:

- integrated attention kernel time
- surrounding layout/cache kernels
- total request prefill time

## Repetition

Run every NCU target at least 3 times and compare medians.

Single-run results are allowed only for debugging.

## Commands

Latency sweep:

```bash
PYTHON_BIN=/home/xuliren/anaconda3/envs/vllm-dev/bin/python \
bash scripts/run_flash_attn_fa2_latency_sweep.sh
```

One NCU point:

```bash
PYTHON_BIN=/home/xuliren/anaconda3/envs/vllm-dev/bin/python \
BATCH=1 SEQ_LEN=8192 RUN_ID=b1_s8192_run1 \
bash scripts/run_flash_attn_fa2_ncu_profile.sh
```
