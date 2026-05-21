#!/bin/bash
set -euo pipefail

# Nsight Compute profile for one same-head FA1/FA2 CUDA forward.

PYTHON_BIN="${PYTHON_BIN:-python}"
BACKEND="${BACKEND:-fa2}"
FA2_INTERFACE="${FA2_INTERFACE:-varlen}"
FLASH_ATTN_SOURCE="${FLASH_ATTN_SOURCE:-}"
REPORT_ROOT="${REPORT_ROOT:-results/traces/ncu/fa1_fa2_same_heads}"
RUN_ID="${RUN_ID:-${BACKEND}_b1_s8192_run1}"
BATCH="${BATCH:-1}"
SEQ_LEN="${SEQ_LEN:-8192}"
Q_HEADS="${Q_HEADS:-32}"
KV_HEADS="${KV_HEADS:-32}"
HEAD_DIM="${HEAD_DIM:-128}"
DTYPE="${DTYPE:-bf16}"
WARMUP="${WARMUP:-3}"
NCU_SET="${NCU_SET:-full}"
KERNEL_NAME="${KERNEL_NAME:-regex:.*flash.*fwd.*}"
LAUNCH_SKIP="${LAUNCH_SKIP:-0}"
LAUNCH_COUNT="${LAUNCH_COUNT:-1}"

if [[ "${Q_HEADS}" != "${KV_HEADS}" ]]; then
  echo "same-head profile requires Q_HEADS == KV_HEADS, got ${Q_HEADS} vs ${KV_HEADS}" >&2
  exit 2
fi

OUT_DIR="${REPORT_ROOT}/${BACKEND}/b${BATCH}_s${SEQ_LEN}/${RUN_ID}"
mkdir -p "${OUT_DIR}"
OUT_PREFIX="${OUT_DIR}/${BACKEND}__same_heads__b${BATCH}__s${SEQ_LEN}__${DTYPE}__${RUN_ID}"

cmd=(
  "${PYTHON_BIN}" scripts/profile_flash_attn_same_heads_once.py
  --backend "${BACKEND}"
  --fa2-interface "${FA2_INTERFACE}"
  --batch "${BATCH}"
  --seq-len "${SEQ_LEN}"
  --heads "${Q_HEADS}"
  --head-dim "${HEAD_DIM}"
  --dtype "${DTYPE}"
  --warmup "${WARMUP}"
)

if [[ -n "${FLASH_ATTN_SOURCE}" ]]; then
  cmd+=(--flash-attn-source "${FLASH_ATTN_SOURCE}")
fi

echo "NCU output prefix: ${OUT_PREFIX}"
echo "Backend: ${BACKEND}"
echo "FA2 interface: ${FA2_INTERFACE}"
echo "Kernel selector: ${KERNEL_NAME}"
echo "Shape: batch=${BATCH}, seq=${SEQ_LEN}, heads=${Q_HEADS}, head_dim=${HEAD_DIM}, dtype=${DTYPE}"

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
