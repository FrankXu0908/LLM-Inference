# PD Routing Experiment Plan

## Goal

This experiment tests whether separating long-prefill traffic from decode-sensitive traffic protects decode latency.

This is not full KV-transfer-based PD disaggregation. It is workload-level PD routing:

```text
A: long-prefill traffic      8192 input / 256 output
B: decode-sensitive traffic   256 input / 8192 output
```

The metric of interest is B-side tail latency:

```text
P99 ITL
P99 TTFT
P99 E2EL
output tok/s
```

## Existing B-Only Ideal

The B-only ideal already exists from the AWQ-Marlin + FP8 KV decode-heavy run:

```text
results/tables/Qwen3-8B/awq_marlin_kv_fp8_dp1_short_prompt_long_output_triton_attn
```

At `c=8`:

| Metric | Value |
|---|---:|
| Output tok/s | `860.91` |
| P99 TTFT | `246.13 ms` |
| P99 ITL | `15.09 ms` |
| P99 E2EL | `76.19 s` |
| KV usage | `30.65%` |
| Waiting | `0` |

This is the decode-sensitive lower bound when B runs without long-prefill interference.

## Line 1: Mixed Baseline

Run one DP=2 endpoint and send A and B traffic to the same endpoint at the same time.

Start server:

```bash
CUDA_VISIBLE_DEVICES=0,1 \
MODEL_CONFIG=configs/qwen3_8b_awq_marlin_kv_fp8_dp2.yaml \
bash scripts/run_server.sh
```

Run mixed A/B benchmark:

```bash
RESULT_DIR=results/tables/Qwen3-8B/pd_routing/mixed_baseline_dp2_a4_b8 \
A_MODEL_CONFIG=configs/qwen3_8b_awq_marlin_kv_fp8_dp2.yaml \
B_MODEL_CONFIG=configs/qwen3_8b_awq_marlin_kv_fp8_dp2.yaml \
A_HOST=127.0.0.1 A_PORT=8000 \
B_HOST=127.0.0.1 B_PORT=8000 \
A_INPUT_LEN=8192 A_OUTPUT_LEN=256 A_NUM_PROMPTS=128 A_CONCURRENCY=4 A_SEED=42 \
B_INPUT_LEN=256 B_OUTPUT_LEN=8192 B_NUM_PROMPTS=32 B_CONCURRENCY=8 B_SEED=43 \
bash scripts/run_mixed_vllm_bench_pair.sh
```

Interpretation:

- A and B share one DP=2 serving endpoint.
- vLLM decides how to distribute mixed traffic across the two data-parallel replicas.
- B can be co-scheduled with A on the same replica, so B-side ITL can show prefill/decode interference.

## Line 2: PD Routing

Run two independent DP=1 endpoints and route A to GPU0, B to GPU1.

Start A / long-prefill endpoint:

```bash
CUDA_VISIBLE_DEVICES=0 \
MODEL_CONFIG=configs/qwen3_8b_awq_marlin_kv_fp8.yaml \
PORT=8000 \
bash scripts/run_server.sh
```

Start B / decode-sensitive endpoint:

```bash
CUDA_VISIBLE_DEVICES=1 \
MODEL_CONFIG=configs/qwen3_8b_awq_marlin_kv_fp8.yaml \
PORT=8001 \
bash scripts/run_server.sh
```

Run routed A/B benchmark:

```bash
RESULT_DIR=results/tables/Qwen3-8B/pd_routing/routed_dp1_a4_b8 \
A_MODEL_CONFIG=configs/qwen3_8b_awq_marlin_kv_fp8.yaml \
B_MODEL_CONFIG=configs/qwen3_8b_awq_marlin_kv_fp8.yaml \
A_HOST=127.0.0.1 A_PORT=8000 \
B_HOST=127.0.0.1 B_PORT=8001 \
A_INPUT_LEN=8192 A_OUTPUT_LEN=256 A_NUM_PROMPTS=128 A_CONCURRENCY=4 A_SEED=42 \
B_INPUT_LEN=256 B_OUTPUT_LEN=8192 B_NUM_PROMPTS=32 B_CONCURRENCY=8 B_SEED=43 \
bash scripts/run_mixed_vllm_bench_pair.sh
```

Interpretation:

- A and B are active at the same time.
- A cannot interfere with B's GPU execution path.
- If B-side P99 ITL moves close to the B-only ideal, routing is effective.

## Decision Rule

The claim is supported if:

```text
Mixed baseline B P99 ITL > PD routing B P99 ITL ~= B-only ideal B P99 ITL
```

For the current B-only ideal, the target is:

```text
B-only ideal c=8 P99 ITL: 15.09 ms
```

The experiment should be reported from B's perspective first. A-side throughput is secondary.

## Artifacts

- Runner: `scripts/run_mixed_vllm_bench_pair.sh`
- DP=2 mixed config: `configs/qwen3_8b_awq_marlin_kv_fp8_dp2.yaml`
- DP=1 routed config: `configs/qwen3_8b_awq_marlin_kv_fp8.yaml`
