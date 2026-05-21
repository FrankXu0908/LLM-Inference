# Kernel-Level Attention Analysis

## Goal

Compare two Triton FlashAttention-style implementations at the kernel level.

Serving metrics answer whether one implementation improves the user-visible result. Kernel profiling answers why.

For this project, use:

- Nsight Systems: timeline, kernel names, kernel duration, launch order.
- Nsight Compute: per-kernel microarchitecture metrics.

## Why Nsight Systems Is Not Enough

Nsight Systems can tell us:

- which kernel is hot
- how long each kernel takes
- whether kernels are fragmented
- whether attention overlaps with other work

But it cannot fully explain:

- whether a kernel is compute-bound or memory-bound
- whether it is limited by DRAM, L2, shared memory, registers, occupancy, or warp stalls
- whether FA1-style or FA2-style tiling improves arithmetic intensity

For FA implementation comparison, Nsight Compute is required.

## Workflow

### 1. Capture Nsight Systems First

For vLLM serving, prefer the official vLLM profiler flow:

- start the server with `--profiler-config.profiler cuda`
- wrap the server with `nsys profile --capture-range=cudaProfilerApi`
- run `vllm bench serve --profile` from a client shell
- stop the server after the client finishes so Nsight Systems flushes the report

Server:

```bash
PROFILER=cuda \
MODEL_CONFIG=configs/qwen3_8b_dense_triton_attn.yaml \
PHASE=prefill \
RUN_ID=baseline_prefill_8192_server \
bash scripts/run_profiled_vllm_server_nsys.sh
```

Client, in another shell:

```bash
PROFILE=true \
MODEL_CONFIG=configs/qwen3_8b_dense_triton_attn.yaml \
RESULT_DIR=results/tables/Qwen3-8B/profiling/triton_prefill_8192 \
CONCURRENCIES="1" \
RANDOM_INPUT_LEN=8192 \
RANDOM_OUTPUT_LEN=1 \
NUM_PROMPTS=2 \
SEED=42 \
TEMPERATURE=0 \
CAPTURE_METRICS=false \
bash scripts/run_vllm_bench_concurrency.sh
```

Then stop the server process.

The older offline single-request script is still useful for quick smoke tests,
but the server/client flow above is the preferred path because it follows
vLLM's profiler trigger mechanism.

### 1.1. Offline Smoke Test

Use the existing single-request trace:

```bash
PYTHON_BIN=/home/xuliren/anaconda3/envs/vllm-dev/bin/python \
MODEL_CONFIG=configs/qwen3_8b_dense_triton_attn.yaml \
PHASE=prefill \
INPUT_TOKENS=8192 \
OUTPUT_TOKENS=1 \
RUN_ID=baseline_prefill_8192 \
bash scripts/run_qwen3_8b_triton_nsys_profile.sh
```

Open the `.nsys-rep` in GUI or parse it:

```bash
python benchmark/analysis/profiling/nsys/parse_nsys.py \
  --trace-file results/traces/nsys/prefill/qwen3_8b_dp1_bf16_triton/baseline_prefill_8192/qwen3_8b__qwen3_8b_dp1_bf16_triton__prefill__c1__in8192__out1__baseline_prefill_8192.nsys-rep \
  --output-dir results/analysis/profiling/nsys/qwen3_8b_triton_prefill_8192
```

Then inspect top kernels:

```bash
python - <<'PY'
import pandas as pd
df = pd.read_csv("results/analysis/profiling/nsys/qwen3_8b_triton_prefill_8192/gpu_kernels.csv")
top = df.groupby("name", as_index=False)["duration_ns"].sum().sort_values("duration_ns", ascending=False).head(30)
print(top.to_string(index=False))
PY
```

Use the real kernel name to set `KERNEL_NAME` for Nsight Compute.

### 2. Capture One Attention Kernel With Nsight Compute

Start with one hot prefill attention kernel:

```bash
PYTHON_BIN=/home/xuliren/anaconda3/envs/vllm-dev/bin/python \
MODEL_CONFIG=configs/qwen3_8b_dense_triton_attn.yaml \
PHASE=prefill \
INPUT_TOKENS=8192 \
OUTPUT_TOKENS=1 \
RUN_ID=baseline_prefill_8192_attn \
KERNEL_NAME="regex:.*triton.*" \
LAUNCH_SKIP=0 \
LAUNCH_COUNT=1 \
GPU_MEMORY_UTILIZATION=0.75 \
ENFORCE_EAGER=true \
WARMUP_TOKENS=0 \
bash scripts/run_qwen3_8b_triton_ncu_profile.sh
```

`KERNEL_NAME="regex:.*triton.*"` is only a starting point. Tighten it after seeing actual kernel names from Nsight Systems.

If Nsight Compute reports `ERR_NVGPUCTRPERM`, GPU performance counters are not
available to the current user. Fix the driver permission first, otherwise NCU
cannot collect roofline/SM/DRAM/warp-stall metrics.

If Nsight Compute OOMs, reduce profiling memory pressure:

```bash
GPU_MEMORY_UTILIZATION=0.70
ENFORCE_EAGER=true
LAUNCH_COUNT=1
```

NCU can require extra memory because it replays and instruments kernels. Do not
use the same high `gpu_memory_utilization=0.95` setting that is used for serving
throughput benchmarks.

For attention-kernel comparison, keep `WARMUP_TOKENS=0` in the NCU run. Otherwise
`LAUNCH_SKIP=0` can capture the small warmup attention kernel instead of the
long-prefill kernel.

For final FA1-vs-baseline claims, repeat each NCU target at least 3 times and
compare medians:

```bash
for r in 1 2 3; do
  PYTHON_BIN=/home/xuliren/anaconda3/envs/vllm-dev/bin/python \
  MODEL_CONFIG=configs/qwen3_8b_dense_triton_attn.yaml \
  PHASE=prefill \
  INPUT_TOKENS=8192 \
  OUTPUT_TOKENS=1 \
  RUN_ID=baseline_prefill_8192_unified_attention_run${r} \
  KERNEL_NAME="regex:kernel_unified_attention_2d" \
  LAUNCH_SKIP=0 \
  LAUNCH_COUNT=1 \
  GPU_MEMORY_UTILIZATION=0.75 \
  ENFORCE_EAGER=true \
  WARMUP_TOKENS=0 \
  bash scripts/run_qwen3_8b_triton_ncu_profile.sh
done
```

Single-run NCU is fine for debugging, but not for final claims. Compare median
duration and median microarchitecture counters.

### 3. Compare Against FA1 Integration

After the external FA1 Triton kernel is wired into vLLM, repeat the same capture:

```bash
MODE=qwen3_8b_dp1_bf16_triton_fa1 \
PHASE=prefill \
INPUT_TOKENS=8192 \
OUTPUT_TOKENS=1 \
RUN_ID=fa1_prefill_8192_attn \
KERNEL_NAME="regex:.*fa1.*|.*triton.*attn.*" \
bash scripts/run_qwen3_8b_triton_ncu_profile.sh
```

Keep the workload, dtype, KV format, and model fixed.

## Metrics To Compare

### Runtime

- Kernel duration
- Number of launches
- Kernel launch fragmentation

### Roofline Position

- Achieved compute throughput
- Achieved memory bandwidth
- Arithmetic intensity
- Compute-bound vs memory-bound classification

### Memory Hierarchy

- DRAM throughput
- L2 hit rate / L2 throughput
- Shared memory usage
- Load/store efficiency

### Occupancy And Scheduling

- Achieved occupancy
- Registers per thread
- Shared memory per block
- Active warps
- Warp stall reasons

Important stall categories:

- memory dependency
- long scoreboard
- not selected
- barrier
- math pipe unavailable

### Tensor / FP Pipe Usage

For BF16 attention, check whether the kernel is using tensor cores effectively or falling back to less efficient instruction paths.

## What FA1 vs FA2-Like Differences Usually Look Like

FA1-style kernels often show:

- simpler tiling
- more memory traffic to intermediate state
- lower arithmetic intensity
- stronger memory dependency stalls at long sequence length

FA2-style kernels usually try to improve:

- work partitioning across warps/blocks
- online softmax scheduling
- non-matmul FLOP overhead
- memory reuse
- occupancy / warp-level parallelism

For our vLLM integration, the most important question is not only whether the standalone kernel is faster. It is whether the kernel still wins after vLLM's KV layout and metadata requirements are included.

## Decision Table

| Observation | Interpretation | Next action |
|---|---|---|
| FA1 kernel duration is lower and no extra layout kernels appear | Promising integration | Run serving A/B |
| FA1 kernel is lower but layout conversion kernels erase gain | Integration mismatch | Optimize layout or stop |
| FA1 has lower occupancy / high register pressure | Kernel implementation issue | Tune block size / num warps / stages |
| FA1 has higher DRAM throughput but same runtime | More bandwidth pressure | Check tiling and KV reads |
| FA1 improves prefill but not decode | Expected if only prefill kernel changed | Keep scope as TTFT/prefill optimization |

## Current Baseline: Qwen3-8B BF16 Triton Prefill

Baseline target:

- Model: `Qwen3-8B`
- dtype: BF16
- backend: vLLM `TRITON_ATTN`
- kernel: `kernel_unified_attention_2d`
- workload: `8192 input / 1 output`
- NCU repetitions: 3

Artifacts:

- `results/traces/ncu/prefill/qwen3_8b_dp1_bf16_triton/baseline_prefill_8192_unified_attention_run*/`
- `results/analysis/profiling/ncu/qwen3_8b_triton_prefill_8192_unified_attention_runs/ncu_runs_summary.csv`
- `results/analysis/profiling/ncu/qwen3_8b_triton_prefill_8192_unified_attention_runs/ncu_runs_summary.json`

Median metrics:

| Metric | Median |
|---|---:|
| Duration | `8.68 ms` |
| Grid Size | `16392` |
| Block Size | `128` |
| Waves / SM | `32.02` |
| Compute throughput | `59.49%` |
| Memory throughput | `16.39 GB/s` |
| DRAM throughput | `1.67%` |
| L2 hit rate | `99.71%` |
| L1/TEX hit rate | `5.89%` |
| SM Busy | `30.20%` |
| Mem Busy | `87.84%` |
| Issue slots busy | `30.20%` |
| No eligible warp | `69.81%` |
| Active warps / scheduler | `3.95` |
| Eligible warps / scheduler | `0.38` |
| Registers / thread | `91` |
| Dynamic shared memory / block | `21.76 KB` |
| Theoretical occupancy | `33.33%` |
| Achieved occupancy | `32.89%` |
| Branch efficiency | `40.03%` |
| Avg. divergent branches | `99712` |

Interpretation:

- The baseline is stable across three NCU runs.
- The kernel is not DRAM-bandwidth limited; DRAM throughput is very low while L2 hit rate is very high.
- The strongest symptoms are scheduler/warp availability and shared-memory behavior: high `No Eligible`, low eligible warps per scheduler, and NCU warnings about shared-memory conflicts / uncoalesced shared access.
- FA1 integration should be compared against this baseline with the same workload and NCU settings.

## Output Layout

Nsight Compute reports should be saved under:

```text
results/traces/ncu/<phase>/<mode>/<run_id>/
```

Nsight Systems reports should be saved under:

```text
results/traces/nsys/<phase>/<mode>/<run_id>/
```
