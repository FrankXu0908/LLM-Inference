#!/bin/bash
set -euo pipefail

# Capture Nsight Compute metrics for Qwen3-8B BF16 TRITON_ATTN.
#
# Recommended workflow:
#   1. Run Nsight Systems first to identify hot attention kernel names.
#   2. Set KERNEL_NAME to a regex matching one target kernel.
#   3. Run this script with a small launch count.
#
# Example:
#   KERNEL_NAME="regex:.*triton.*attn.*" \
#   PHASE=prefill INPUT_TOKENS=8192 OUTPUT_TOKENS=1 \
#   RUN_ID=baseline_prefill_8192_attn \
#   bash scripts/run_qwen3_8b_triton_ncu_profile.sh

MODEL_CONFIG="${MODEL_CONFIG:-configs/qwen3_8b_dense_triton_attn.yaml}"
PYTHON_BIN="${PYTHON_BIN:-python}"
PHASE="${PHASE:-prefill}"
INPUT_TOKENS="${INPUT_TOKENS:-8192}"
OUTPUT_TOKENS="${OUTPUT_TOKENS:-1}"
RUN_ID="${RUN_ID:-run1}"
MODE="${MODE:-qwen3_8b_dp1_bf16_triton}"
REPORT_ROOT="${REPORT_ROOT:-results/traces/ncu}"
KERNEL_NAME="${KERNEL_NAME:-regex:.*}"
LAUNCH_SKIP="${LAUNCH_SKIP:-0}"
LAUNCH_COUNT="${LAUNCH_COUNT:-1}"
NCU_SET="${NCU_SET:-full}"
MAX_NUM_SEQS="${MAX_NUM_SEQS:-1}"
GPU_MEMORY_UTILIZATION="${GPU_MEMORY_UTILIZATION:-0.75}"
ENFORCE_EAGER="${ENFORCE_EAGER:-true}"
WARMUP_TOKENS="${WARMUP_TOKENS:-0}"
VLLM_ENABLE_V1_MULTIPROCESSING="${VLLM_ENABLE_V1_MULTIPROCESSING:-0}"
VLLM_WORKER_MULTIPROC_METHOD="${VLLM_WORKER_MULTIPROC_METHOD:-spawn}"

OUT_DIR="${REPORT_ROOT}/${PHASE}/${MODE}/${RUN_ID}"
mkdir -p "${OUT_DIR}"

OUT_PREFIX="${OUT_DIR}/qwen3_8b__${MODE}__${PHASE}__c1__in${INPUT_TOKENS}__out${OUTPUT_TOKENS}__${RUN_ID}"

cmd=(
  "${PYTHON_BIN}" scripts/profile_qwen3_8b_triton_once.py
  --model-config "${MODEL_CONFIG}"
  --phase "${PHASE}"
  --input-tokens "${INPUT_TOKENS}"
  --output-tokens "${OUTPUT_TOKENS}"
  --max-num-seqs "${MAX_NUM_SEQS}"
  --warmup-tokens "${WARMUP_TOKENS}"
)

if [[ -n "${GPU_MEMORY_UTILIZATION}" ]]; then
  cmd+=(--gpu-memory-utilization "${GPU_MEMORY_UTILIZATION}")
fi
if [[ "${ENFORCE_EAGER}" == "true" ]]; then
  cmd+=(--enforce-eager)
fi

echo "NCU output prefix: ${OUT_PREFIX}"
echo "Kernel selector: ${KERNEL_NAME}"
echo "Launch skip/count: ${LAUNCH_SKIP}/${LAUNCH_COUNT}"
echo "Workload: phase=${PHASE}, input=${INPUT_TOKENS}, output=${OUTPUT_TOKENS}, mode=${MODE}"
echo "NCU-safe runtime: gpu_memory_utilization=${GPU_MEMORY_UTILIZATION}, enforce_eager=${ENFORCE_EAGER}"
echo "Warmup tokens: ${WARMUP_TOKENS}"
echo "vLLM multiprocessing: VLLM_ENABLE_V1_MULTIPROCESSING=${VLLM_ENABLE_V1_MULTIPROCESSING}, VLLM_WORKER_MULTIPROC_METHOD=${VLLM_WORKER_MULTIPROC_METHOD}"

export VLLM_ENABLE_V1_MULTIPROCESSING
export VLLM_WORKER_MULTIPROC_METHOD

ncu \
  --target-processes all \
  --set "${NCU_SET}" \
  --kernel-name "${KERNEL_NAME}" \
  --launch-skip "${LAUNCH_SKIP}" \
  --launch-count "${LAUNCH_COUNT}" \
  --force-overwrite \
  --export "${OUT_PREFIX}" \
  "${cmd[@]}"

echo "Saved: ${OUT_PREFIX}.ncu-rep"
