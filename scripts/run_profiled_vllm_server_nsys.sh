#!/bin/bash
set -euo pipefail

# Start a vLLM server under Nsight Systems using vLLM's official profiler flow.
#
# Server side:
#   PROFILER=cuda MODEL_CONFIG=configs/qwen3_8b_dense_triton_attn.yaml \
#     RUN_ID=baseline_prefill_server \
#     bash scripts/run_profiled_vllm_server_nsys.sh
#
# Client side, from another shell:
#   PROFILE=true CONCURRENCIES="1" RANDOM_INPUT_LEN=8192 RANDOM_OUTPUT_LEN=1 \
#     NUM_PROMPTS=2 CAPTURE_METRICS=false \
#     RESULT_DIR=results/tables/Qwen3-8B/profiling/prefill_8192 \
#     bash scripts/run_vllm_bench_concurrency.sh
#
# Stop the server after the benchmark. Nsight Systems writes the .nsys-rep when
# the profiled server process exits.

MODEL_CONFIG="${MODEL_CONFIG:-configs/qwen3_8b_dense_triton_attn.yaml}"
RUN_ID="${RUN_ID:-run1}"
PHASE="${PHASE:-prefill}"
MODE="${MODE:-qwen3_8b_dp1_bf16_triton}"
TRACE_ROOT="${TRACE_ROOT:-results/traces/nsys}"
PROFILER="${PROFILER:-cuda}"
NSYS_BIN="${NSYS_BIN:-nsys}"
VLLM_BIN="${VLLM_BIN:-vllm}"

OUT_DIR="${TRACE_ROOT}/${PHASE}/${MODE}/${RUN_ID}"
mkdir -p "${OUT_DIR}"
OUT_PREFIX="${OUT_DIR}/qwen3_8b__${MODE}__${PHASE}__server__${RUN_ID}"

echo "Starting profiled vLLM server"
echo "Trace output prefix: ${OUT_PREFIX}"
echo "MODEL_CONFIG=${MODEL_CONFIG}"
echo "PROFILER=${PROFILER}"
echo "VLLM_BIN=${VLLM_BIN}"
echo
echo "Run the benchmark client in another shell with PROFILE=true."
echo "Kill this server after the client finishes so Nsight can flush the report."

PROFILER="${PROFILER}" \
VLLM_BIN="${VLLM_BIN}" \
"${NSYS_BIN}" profile \
  --trace-fork-before-exec=true \
  --cuda-graph-trace=node \
  --trace=cuda,nvtx,osrt \
  --capture-range=cudaProfilerApi \
  --capture-range-end=repeat \
  --force-overwrite=true \
  "--output=${OUT_PREFIX}" \
  bash scripts/run_server.sh
