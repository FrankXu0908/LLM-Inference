# FA2 Backend Tuning Plan

This note is the working guide for the next kernel-level stage:

> Use profiling to locate the FA2 bottleneck, modify one backend parameter at a
> time, and verify whether the hypothesis is actually true.

The goal is not to produce a generic FlashAttention benchmark. The goal is to
explain, for Qwen3-8B on RTX 4090, which FA2 backend choices matter and which
ones are dead ends.

## Fixed Baseline

Use the same workload family as the FA1 vs FA2 study:

| Field | Value |
|---|---:|
| model shape source | `Qwen3-8B` |
| GPU | RTX 4090 / Ada / SM89 |
| dtype | BF16 |
| attention | causal |
| dropout | false |
| alibi / local / softcap | false |
| head dim | `128` |
| same-head comparison | Q/K/V heads = `32` |
| primary long-context seq | `8192` |
| primary batch | `1`, `4`, `16` |

Current FA2 CUDA dispatch observed from source and NCU demangled kernel names:

```text
Flash_fwd_kernel_traits<128, 64, 64, 4, false, false, bf16>
```

For this target, treat the current baseline as:

```text
BM = 64
BN = 64
kNWarps = 4
```

Important: the `4` above is `kNWarps` in `Flash_fwd_kernel_traits`, not
automatically `num_stages`. Before claiming a `num_stages` experiment, locate
the actual pipeline / stage control in the CUDA source and record the exact diff.

## Existing Evidence

Completed measurements:

- `fa2_cuda_baseline_results.md`
- `fa1_vs_fa2_same_head_observations.md`
- `results/analysis/profiling/ncu/fa2_cuda_standalone_summary/fa2_ncu_matrix_summary.csv`
- `results/analysis/profiling/ncu/fa1_fa2_same_heads/same_heads_fa1_fa2_compare.csv`

Key observation:

- FA1 and FA2 are both tiled IO-aware attention implementations.
- In our same-head BF16 causal test, FA2 is faster because it shows much lower
  DRAM traffic, much higher L2 hit rate, many more waves per SM, and shifts the
  bottleneck from memory-scoreboard pressure toward execution-pipe wait.
- Bigger batch alone does not remove the FA2 internal limiter. At `seq=8192`,
  batch `16 -> 32 -> 64` increases waves, but SM utilization and TFLOP/s are
  already near a plateau.

## Hypotheses To Verify

### H1: Tile Shape Controls the Latency / Utilization Tradeoff

Baseline:

```text
BM64 BN64 warps4
```

Candidate variants:

| Variant | Why test it | Expected risk |
|---|---|---|
| `BM64 BN128 warps4` | Emulates a larger N tile and the user's FA1-style `BLOCK_N=128` hypothesis | May reduce parallel waves, increase per-CTA work, and worsen latency |
| `BM128 BN64 warps4` | Tests whether larger M tile improves matmul efficiency | May increase register/shared-memory pressure or reduce occupancy |
| `BM64 BN32 warps4` | Tests whether smaller N tile improves scheduling and tail behavior | May increase overhead and reduce reuse |

Decision rule:

- Better only if median latency improves and NCU does not show worse correctness,
  worse occupancy/eligible-warps collapse, or much higher memory traffic.

### H2: Pipeline / Stage Control May Hide Memory Dependency Better

Do not assume `Flash_fwd_kernel_traits<..., 4, ...>` is `num_stages`.

First source task:

```text
Find the actual pipeline/staging knob used by the FA2 CUDA path for BF16 hdim128 causal SM89.
```

Then test a small set of stage variants, for example:

| Variant | Requirement |
|---|---|
| baseline stage count | record exact source location |
| lower stage count | verify lower smem/register pressure does not hurt latency |
| higher stage count | verify additional buffering hides dependency without spilling |

Decision rule:

- A stage change is useful only if it increases eligible warps / issue activity
  or lowers top stalls without increasing latency.

### H3: `BLOCK_N=128` Is Probably a Regression, But We Should Measure It

The `BN128` test is valuable even if it regresses, because it answers a concrete
backend-design question:

```text
Does a larger N tile help this RTX 4090 BF16 causal head_dim=128 workload,
or does it reduce parallelism / increase resource pressure?
```

Expected signs of regression:

- higher kernel duration
- fewer waves per SM
- lower eligible warps per scheduler
- higher registers/thread or shared memory/block
- more severe execution dependency stalls

## Required Measurement Table

Every backend variant should produce one row per important workload.

| variant | batch | seq | median latency ms | NCU duration ms | SM % | DRAM % | est DRAM GB | L2 hit % | regs/thread | smem/block KB | achieved occ % | eligible warps/sched | issue active % | waves/SM | top stall | decision |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---|---|
| baseline BM64 BN64 W4 | 1 | 8192 | TBD | TBD | TBD | TBD | TBD | TBD | TBD | TBD | TBD | TBD | TBD | TBD | TBD | baseline |
| baseline BM64 BN64 W4 | 16 | 8192 | TBD | TBD | TBD | TBD | TBD | TBD | TBD | TBD | TBD | TBD | TBD | TBD | TBD | baseline |
| BM64 BN128 W4 | 1 | 8192 | TBD | TBD | TBD | TBD | TBD | TBD | TBD | TBD | TBD | TBD | TBD | TBD | TBD | test |
| BM64 BN128 W4 | 16 | 8192 | TBD | TBD | TBD | TBD | TBD | TBD | TBD | TBD | TBD | TBD | TBD | TBD | TBD | test |

## Correctness Gate

Before running NCU, each modified kernel must pass a numerical check against the
baseline FA2 implementation.

Record:

- max absolute error
- max relative error
- whether the output tolerance is acceptable for BF16
- whether the variant changes any layout or API behavior

No correctness gate, no performance claim.

## Profiling Workflow

1. Patch one FA2 backend variable.
2. Rebuild / reinstall the local FlashAttention package in the target conda env.
3. Run the latency sweep first.
4. Run NCU only for the shapes that changed meaningfully.
5. Save source diff, benchmark JSON/CSV, and NCU CSV/JSON together.

Suggested latency command shape:

```bash
PYTHON_BIN=/home/xuliren/anaconda3/envs/deeplearning/bin/python \
BACKEND=fa2 \
FA2_INTERFACE=varlen \
BATCHES="1 4 16" \
SEQ_LENS="8192" \
REPEATS=30 \
WARMUP=10 \
RUN_ID=<variant_name> \
bash scripts/run_flash_attn_same_heads_latency_sweep.sh
```

Suggested NCU command shape:

```bash
PYTHON_BIN=/home/xuliren/anaconda3/envs/deeplearning/bin/python \
BACKEND=fa2 \
FA2_INTERFACE=varlen \
BATCH=16 \
SEQ_LEN=8192 \
RUN_ID=<variant_name> \
KERNEL_NAME="regex:.*flash.*fwd.*" \
LAUNCH_SKIP=0 \
LAUNCH_COUNT=1 \
bash scripts/run_flash_attn_same_heads_ncu_profile.sh
```

## Reporting Rule

Each tuning result should answer:

```text
What changed?
What did profiling predict?
What happened to latency?
What happened to SM/DRAM/occupancy/eligible warps/stalls?
Do we keep it, reject it, or narrow the search?
```

This keeps the work grounded in measured bottlenecks instead of intuition.
