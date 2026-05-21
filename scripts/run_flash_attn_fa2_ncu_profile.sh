#!/bin/bash
set -euo pipefail

# Nsight Compute profile for one standalone FlashAttention-2 CUDA forward.
#
# Start with one representative point, then repeat interesting points:
#   BATCH=1 SEQ_LEN=8192 RUN_ID=b1_s8192 \
#   bash scripts/run_flash_attn_fa2_ncu_profile.sh

PYTHON_BIN="${PYTHON_BIN:-python}"
FLASH_ATTN_SOURCE="${FLASH_ATTN_SOURCE:-/home/xuliren/repo/flash-attention}"
REPORT_ROOT="${REPORT_ROOT:-results/traces/ncu/fa2_cuda_standalone}"
RUN_ID="${RUN_ID:-b1_s8192_run1}"
BATCH="${BATCH:-1}"
SEQ_LEN="${SEQ_LEN:-8192}"
Q_HEADS="${Q_HEADS:-32}"
KV_HEADS="${KV_HEADS:-8}"
HEAD_DIM="${HEAD_DIM:-128}"
DTYPE="${DTYPE:-bf16}"
WARMUP="${WARMUP:-3}"
NCU_SET="${NCU_SET:-full}"
KERNEL_NAME="${KERNEL_NAME:-regex:.*flash.*fwd.*}"
LAUNCH_SKIP="${LAUNCH_SKIP:-0}"
LAUNCH_COUNT="${LAUNCH_COUNT:-1}"

OUT_DIR="${REPORT_ROOT}/b${BATCH}_s${SEQ_LEN}/${RUN_ID}"
mkdir -p "${OUT_DIR}"
OUT_PREFIX="${OUT_DIR}/fa2_cuda__b${BATCH}__s${SEQ_LEN}__${DTYPE}__${RUN_ID}"

echo "NCU output prefix: ${OUT_PREFIX}"
echo "Kernel selector: ${KERNEL_NAME}"
echo "Shape: batch=${BATCH}, seq=${SEQ_LEN}, q_heads=${Q_HEADS}, kv_heads=${KV_HEADS}, head_dim=${HEAD_DIM}, dtype=${DTYPE}"

ncu \
  --target-processes all \
  --set "${NCU_SET}" \
  --kernel-name "${KERNEL_NAME}" \
  --launch-skip "${LAUNCH_SKIP}" \
  --launch-count "${LAUNCH_COUNT}" \
  --force-overwrite \
  --export "${OUT_PREFIX}" \
  "${PYTHON_BIN}" scripts/profile_flash_attn_fa2_once.py \
    --flash-attn-source "${FLASH_ATTN_SOURCE}" \
    --batch "${BATCH}" \
    --seq-len "${SEQ_LEN}" \
    --q-heads "${Q_HEADS}" \
    --kv-heads "${KV_HEADS}" \
    --head-dim "${HEAD_DIM}" \
    --dtype "${DTYPE}" \
    --warmup "${WARMUP}"

echo "Saved: ${OUT_PREFIX}.ncu-rep"
