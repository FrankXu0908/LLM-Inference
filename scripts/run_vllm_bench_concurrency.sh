#!/bin/bash
set -euo pipefail

# Run a burst-arrival max-concurrency sweep with vLLM's native benchmark client.
#
# Example, Baseline B short context:
#   RESULT_DIR=results/tables/Qwen3-8B/baseline_b_dp2_standard \
#   CONCURRENCIES="1 4 8 16" RANDOM_INPUT_LEN=512 RANDOM_OUTPUT_LEN=256 \
#   bash scripts/run_vllm_bench_concurrency.sh
#
# Example, Baseline B long context:
#   RESULT_DIR=results/tables/Qwen3-8B/baseline_b_dp2_long_context \
#   CONCURRENCIES="1 2 4 8" RANDOM_INPUT_LEN=8192 RANDOM_OUTPUT_LEN=256 \
#   NUM_PROMPTS=128 SEED=42 TEMPERATURE=0 \
#   bash scripts/run_vllm_bench_concurrency.sh
#
# Example, start the AWQ-Marlin server before running AWQ benchmark sweeps:
#   vllm serve /home/xuliren/repo/models/Qwen/Qwen3-8B-AWQ \
#     --served-model-name Qwen3-8B-AWQ-Marlin \
#     --dtype half \
#     --quantization awq_marlin \
#     --max-model-len 10000 \
#     --gpu-memory-utilization 0.85 \
#     --max-num-seqs 64 \
#     --enable-prefix-caching \
#     --enable-chunked-prefill
#
# Example, AWQ-Marlin short-context sweep:
#   MODEL_CONFIG=configs/qwen3_8b_awq_marlin.yaml \
#   RESULT_DIR=results/tables/Qwen3-8B/awq_marlin_dp1_standard \
#   CONCURRENCIES="1 4 8 16" RANDOM_INPUT_LEN=512 RANDOM_OUTPUT_LEN=256 \
#   NUM_PROMPTS=256 SEED=42 TEMPERATURE=0 \
#   bash scripts/run_vllm_bench_concurrency.sh

MODEL_CONFIG="${MODEL_CONFIG:-configs/qwen3_8b_dense.yaml}"
CONCURRENCIES="${CONCURRENCIES:-1 4 8 16}"
RESULT_DIR="${RESULT_DIR:-results/tables/Qwen3-8B/vllm_bench/concurrency}"
VLLM_BIN="${VLLM_BIN:-vllm}"
CAPTURE_METRICS="${CAPTURE_METRICS:-true}"
METRICS_INTERVAL="${METRICS_INTERVAL:-1}"
PROFILE="${PROFILE:-false}"

readarray -t CFG < <(python - "$MODEL_CONFIG" <<'PY'
import sys, yaml
cfg = yaml.safe_load(open(sys.argv[1], "r", encoding="utf-8")) or {}
model = cfg.get("model")
served = cfg.get("served_model_name")
host = cfg.get("host", "127.0.0.1")
port = cfg.get("port", 8000)
print(model or "")
print(served or model or "")
print(host)
print(port)
PY
)

MODEL_PATH="${MODEL_PATH:-${CFG[0]}}"
SERVED_MODEL="${SERVED_MODEL:-${CFG[1]}}"
HOST="${HOST:-${CFG[2]}}"
PORT="${PORT:-${CFG[3]}}"

RANDOM_INPUT_LEN="${RANDOM_INPUT_LEN:-512}"
RANDOM_OUTPUT_LEN="${RANDOM_OUTPUT_LEN:-256}"
NUM_PROMPTS="${NUM_PROMPTS:-256}"
TEMPERATURE="${TEMPERATURE:-0}"
SEED="${SEED:-42}"

if [[ -z "${MODEL_PATH}" ]]; then
  echo "model is missing in ${MODEL_CONFIG}"
  exit 1
fi

if [[ "${HOST}" == "0.0.0.0" ]]; then
  HOST="127.0.0.1"
fi

mkdir -p "${RESULT_DIR}"

echo "Model path: ${MODEL_PATH}"
echo "Served model: ${SERVED_MODEL}"
echo "Endpoint: ${HOST}:${PORT}"
echo "Dataset: random, input=${RANDOM_INPUT_LEN}, output=${RANDOM_OUTPUT_LEN}, prompts=${NUM_PROMPTS}"
echo "Sampling: temperature=${TEMPERATURE}, seed=${SEED}, random_range_ratio=0"
echo "Concurrency sweep: ${CONCURRENCIES}"
echo "Result dir: ${RESULT_DIR}"
echo "Capture metrics: ${CAPTURE_METRICS}, interval=${METRICS_INTERVAL}s"
echo "vLLM profile trigger: ${PROFILE}"

for c in ${CONCURRENCIES}; do
  echo "===== concurrency=${c} ====="

  nvidia-smi dmon -s pucm -d 1 -o DT > "${RESULT_DIR}/c${c}_dmon.log" &
  MON_PID=$!

  METRICS_PID=""
  if [[ "${CAPTURE_METRICS}" == "true" ]]; then
    OUT="${RESULT_DIR}/c${c}_metrics.prom" \
    HOST="${HOST}" \
    PORT="${PORT}" \
    INTERVAL="${METRICS_INTERVAL}" \
      bash scripts/capture_vllm_metrics.sh &
    METRICS_PID=$!
  fi

  set +e
  cmd=(
    "${VLLM_BIN}" bench serve
    --backend vllm
    --model "${MODEL_PATH}"
    --served-model-name "${SERVED_MODEL}"
    --dataset-name random
    --random-input-len "${RANDOM_INPUT_LEN}"
    --random-output-len "${RANDOM_OUTPUT_LEN}"
    --random-range-ratio 0
    --num-prompts "${NUM_PROMPTS}"
    --max-concurrency "${c}"
    --temperature "${TEMPERATURE}"
    --seed "${SEED}"
    --percentile-metrics ttft,tpot,itl,e2el
    --metric-percentiles 50,90,95,99
    --host "${HOST}"
    --port "${PORT}"
    --save-result
    --result-dir "${RESULT_DIR}"
    --result-filename "c${c}_bench.json"
  )
  if [[ "${PROFILE}" == "true" ]]; then
    cmd+=(--profile)
  fi
  "${cmd[@]}" | tee "${RESULT_DIR}/c${c}_bench.log"
  STATUS=${PIPESTATUS[0]}
  set -e

  kill "${MON_PID}" 2>/dev/null || true
  wait "${MON_PID}" 2>/dev/null || true
  if [[ -n "${METRICS_PID}" ]]; then
    kill "${METRICS_PID}" 2>/dev/null || true
    wait "${METRICS_PID}" 2>/dev/null || true
  fi

  if [[ "${STATUS}" -ne 0 ]]; then
    echo "Benchmark failed for concurrency=${c} with status=${STATUS}" >&2
    exit "${STATUS}"
  fi
done
