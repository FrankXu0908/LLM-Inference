#!/bin/bash
set -euo pipefail

# Standalone FlashAttention-2 CUDA latency sweep for Qwen3-8B attention shape.
#
# This does not start vLLM. It calls flash_attn.flash_attn_func directly.
# Build flash-attn first if import fails:
#   cd /home/xuliren/repo/flash-attention
#   pip install -e . --no-build-isolation

PYTHON_BIN="${PYTHON_BIN:-python}"
FLASH_ATTN_SOURCE="${FLASH_ATTN_SOURCE:-/home/xuliren/repo/flash-attention}"
OUTPUT_JSON="${OUTPUT_JSON:-results/tables/Qwen3-8B/fa2_cuda_standalone/fa2_latency_sweep.json}"
BATCHES="${BATCHES:-1 4 16}"
SEQ_LENS="${SEQ_LENS:-512 2048 8192 16384}"
Q_HEADS="${Q_HEADS:-32}"
KV_HEADS="${KV_HEADS:-8}"
HEAD_DIM="${HEAD_DIM:-128}"
DTYPE="${DTYPE:-bf16}"
WARMUP="${WARMUP:-5}"
ITERS="${ITERS:-20}"
SEED="${SEED:-42}"

"${PYTHON_BIN}" scripts/benchmark_flash_attn_fa2.py \
  --flash-attn-source "${FLASH_ATTN_SOURCE}" \
  --output-json "${OUTPUT_JSON}" \
  --batches ${BATCHES} \
  --seq-lens ${SEQ_LENS} \
  --q-heads "${Q_HEADS}" \
  --kv-heads "${KV_HEADS}" \
  --head-dim "${HEAD_DIM}" \
  --dtype "${DTYPE}" \
  --warmup "${WARMUP}" \
  --iters "${ITERS}" \
  --seed "${SEED}"
