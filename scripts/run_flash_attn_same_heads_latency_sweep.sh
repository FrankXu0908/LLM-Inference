#!/bin/bash
set -euo pipefail

# Same-head FA1/FA2 standalone latency sweep.
# Use Q_HEADS=KV_HEADS=32 for fair FA1 vs FA2 comparison.

PYTHON_BIN="${PYTHON_BIN:-python}"
BACKEND="${BACKEND:-fa2}"
FA2_INTERFACE="${FA2_INTERFACE:-varlen}"
FLASH_ATTN_SOURCE="${FLASH_ATTN_SOURCE:-}"
OUTPUT_JSON="${OUTPUT_JSON:-results/tables/Qwen3-8B/fa1_fa2_same_heads/${BACKEND}_latency_sweep.json}"
BATCHES="${BATCHES:-1 4 16}"
SEQ_LENS="${SEQ_LENS:-512 2048 8192}"
Q_HEADS="${Q_HEADS:-32}"
KV_HEADS="${KV_HEADS:-32}"
HEAD_DIM="${HEAD_DIM:-128}"
DTYPE="${DTYPE:-bf16}"
WARMUP="${WARMUP:-5}"
ITERS="${ITERS:-20}"
SEED="${SEED:-42}"

if [[ "${Q_HEADS}" != "${KV_HEADS}" ]]; then
  echo "same-head benchmark requires Q_HEADS == KV_HEADS, got ${Q_HEADS} vs ${KV_HEADS}" >&2
  exit 2
fi

cmd=(
  "${PYTHON_BIN}" scripts/benchmark_flash_attn_same_heads.py
  --backend "${BACKEND}"
  --fa2-interface "${FA2_INTERFACE}"
  --output-json "${OUTPUT_JSON}"
  --batches ${BATCHES}
  --seq-lens ${SEQ_LENS}
  --heads "${Q_HEADS}"
  --head-dim "${HEAD_DIM}"
  --dtype "${DTYPE}"
  --warmup "${WARMUP}"
  --iters "${ITERS}"
  --seed "${SEED}"
)

if [[ -n "${FLASH_ATTN_SOURCE}" ]]; then
  cmd+=(--flash-attn-source "${FLASH_ATTN_SOURCE}")
fi

"${cmd[@]}"
