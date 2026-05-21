#!/bin/bash
set -euo pipefail

# Capture one Qwen3-8B BF16 TRITON_ATTN request with Nsight Systems.
#
# Example prefill-oriented capture:
#   PHASE=prefill INPUT_TOKENS=8192 OUTPUT_TOKENS=1 \
#     bash scripts/run_qwen3_8b_triton_nsys_profile.sh
#
# Example decode-oriented capture:
#   PHASE=decode INPUT_TOKENS=256 OUTPUT_TOKENS=8192 \
#     bash scripts/run_qwen3_8b_triton_nsys_profile.sh

MODEL_CONFIG="${MODEL_CONFIG:-configs/qwen3_8b_dense_triton_attn.yaml}"
PYTHON_BIN="${PYTHON_BIN:-python}"
PHASE="${PHASE:-prefill}"
INPUT_TOKENS="${INPUT_TOKENS:-8192}"
OUTPUT_TOKENS="${OUTPUT_TOKENS:-1}"
RUN_ID="${RUN_ID:-run1}"
MODE="${MODE:-qwen3_8b_dp1_bf16_triton}"
TRACE_ROOT="${TRACE_ROOT:-results/traces/nsys}"
GPU_MEMORY_UTILIZATION="${GPU_MEMORY_UTILIZATION:-}"
MAX_NUM_SEQS="${MAX_NUM_SEQS:-1}"
ENFORCE_EAGER="${ENFORCE_EAGER:-false}"
VLLM_ENABLE_V1_MULTIPROCESSING="${VLLM_ENABLE_V1_MULTIPROCESSING:-0}"
VLLM_WORKER_MULTIPROC_METHOD="${VLLM_WORKER_MULTIPROC_METHOD:-spawn}"

OUT_DIR="${TRACE_ROOT}/${PHASE}/${MODE}/${RUN_ID}"
mkdir -p "${OUT_DIR}"

OUT_PREFIX="${OUT_DIR}/qwen3_8b__${MODE}__${PHASE}__c1__in${INPUT_TOKENS}__out${OUTPUT_TOKENS}__${RUN_ID}"

cmd=(
  "${PYTHON_BIN}" scripts/profile_qwen3_8b_triton_once.py
  --model-config "${MODEL_CONFIG}"
  --phase "${PHASE}"
  --input-tokens "${INPUT_TOKENS}"
  --output-tokens "${OUTPUT_TOKENS}"
  --max-num-seqs "${MAX_NUM_SEQS}"
)

if [[ -n "${GPU_MEMORY_UTILIZATION}" ]]; then
  cmd+=(--gpu-memory-utilization "${GPU_MEMORY_UTILIZATION}")
fi
if [[ "${ENFORCE_EAGER}" == "true" ]]; then
  cmd+=(--enforce-eager)
fi

echo "Trace output prefix: ${OUT_PREFIX}"
echo "Workload: phase=${PHASE}, input=${INPUT_TOKENS}, output=${OUTPUT_TOKENS}, mode=${MODE}"
echo "vLLM multiprocessing: VLLM_ENABLE_V1_MULTIPROCESSING=${VLLM_ENABLE_V1_MULTIPROCESSING}, VLLM_WORKER_MULTIPROC_METHOD=${VLLM_WORKER_MULTIPROC_METHOD}"

export VLLM_ENABLE_V1_MULTIPROCESSING
export VLLM_WORKER_MULTIPROC_METHOD

nsys profile \
  --trace=cuda,nvtx,osrt \
  --capture-range=cudaProfilerApi \
  --capture-range-end=stop \
  --force-overwrite=true \
  "--output=${OUT_PREFIX}" \
  "${cmd[@]}"

echo "Saved: ${OUT_PREFIX}.nsys-rep"
