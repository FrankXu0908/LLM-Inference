#!/bin/bash
set -euo pipefail

# Run two vLLM benchmark workloads concurrently.
#
# This is used for mixed-workload / PD-routing experiments:
# - Workload A: long-prefill traffic, e.g. 8192 input / 256 output.
# - Workload B: decode-sensitive traffic, e.g. 256 input / 8192 output.
#
# Mixed baseline:
#   A_HOST=127.0.0.1 A_PORT=8000 B_HOST=127.0.0.1 B_PORT=8000 ...
#
# PD routing:
#   A_HOST=127.0.0.1 A_PORT=8000 B_HOST=127.0.0.1 B_PORT=8001 ...

RESULT_DIR="${RESULT_DIR:-results/tables/Qwen3-8B/pd_routing/mixed_pair}"
VLLM_BIN="${VLLM_BIN:-vllm}"
CAPTURE_METRICS="${CAPTURE_METRICS:-true}"
METRICS_INTERVAL="${METRICS_INTERVAL:-1}"

A_MODEL_CONFIG="${A_MODEL_CONFIG:-configs/qwen3_8b_awq_marlin_kv_fp8.yaml}"
B_MODEL_CONFIG="${B_MODEL_CONFIG:-${A_MODEL_CONFIG}}"

A_HOST="${A_HOST:-127.0.0.1}"
A_PORT="${A_PORT:-8000}"
B_HOST="${B_HOST:-127.0.0.1}"
B_PORT="${B_PORT:-8000}"

A_INPUT_LEN="${A_INPUT_LEN:-8192}"
A_OUTPUT_LEN="${A_OUTPUT_LEN:-256}"
A_NUM_PROMPTS="${A_NUM_PROMPTS:-128}"
A_CONCURRENCY="${A_CONCURRENCY:-4}"
A_SEED="${A_SEED:-42}"
A_TEMPERATURE="${A_TEMPERATURE:-0}"

B_INPUT_LEN="${B_INPUT_LEN:-256}"
B_OUTPUT_LEN="${B_OUTPUT_LEN:-8192}"
B_NUM_PROMPTS="${B_NUM_PROMPTS:-32}"
B_CONCURRENCY="${B_CONCURRENCY:-4}"
B_SEED="${B_SEED:-43}"
B_TEMPERATURE="${B_TEMPERATURE:-0}"

read_config() {
  python - "$1" <<'PY'
import sys, yaml
cfg = yaml.safe_load(open(sys.argv[1], "r", encoding="utf-8")) or {}
model = cfg.get("model", "")
served = cfg.get("served_model_name") or model
print(model)
print(served)
PY
}

readarray -t A_CFG < <(read_config "${A_MODEL_CONFIG}")
readarray -t B_CFG < <(read_config "${B_MODEL_CONFIG}")
A_MODEL="${A_MODEL:-${A_CFG[0]}}"
A_SERVED_MODEL="${A_SERVED_MODEL:-${A_CFG[1]}}"
B_MODEL="${B_MODEL:-${B_CFG[0]}}"
B_SERVED_MODEL="${B_SERVED_MODEL:-${B_CFG[1]}}"

mkdir -p "${RESULT_DIR}/a_long_prefill" "${RESULT_DIR}/b_decode_sensitive"

echo "Result dir: ${RESULT_DIR}"
echo "A endpoint: ${A_HOST}:${A_PORT}, model=${A_SERVED_MODEL}, input/output=${A_INPUT_LEN}/${A_OUTPUT_LEN}, prompts=${A_NUM_PROMPTS}, concurrency=${A_CONCURRENCY}"
echo "B endpoint: ${B_HOST}:${B_PORT}, model=${B_SERVED_MODEL}, input/output=${B_INPUT_LEN}/${B_OUTPUT_LEN}, prompts=${B_NUM_PROMPTS}, concurrency=${B_CONCURRENCY}"

nvidia-smi dmon -s pucm -d 1 -o DT > "${RESULT_DIR}/mixed_dmon.log" &
DMON_PID=$!

METRICS_PIDS=()
if [[ "${CAPTURE_METRICS}" == "true" ]]; then
  OUT="${RESULT_DIR}/a_long_prefill/metrics.prom" HOST="${A_HOST}" PORT="${A_PORT}" INTERVAL="${METRICS_INTERVAL}" \
    bash scripts/capture_vllm_metrics.sh &
  METRICS_PIDS+=($!)

  # Avoid double-scraping the same endpoint in mixed-baseline mode.
  if [[ "${A_HOST}:${A_PORT}" != "${B_HOST}:${B_PORT}" ]]; then
    OUT="${RESULT_DIR}/b_decode_sensitive/metrics.prom" HOST="${B_HOST}" PORT="${B_PORT}" INTERVAL="${METRICS_INTERVAL}" \
      bash scripts/capture_vllm_metrics.sh &
    METRICS_PIDS+=($!)
  fi
fi

cleanup() {
  kill "${DMON_PID}" 2>/dev/null || true
  wait "${DMON_PID}" 2>/dev/null || true
  for pid in "${METRICS_PIDS[@]}"; do
    kill "${pid}" 2>/dev/null || true
    wait "${pid}" 2>/dev/null || true
  done
}
trap cleanup EXIT

run_bench() {
  local label="$1"
  local out_dir="$2"
  local host="$3"
  local port="$4"
  local model="$5"
  local served="$6"
  local input_len="$7"
  local output_len="$8"
  local prompts="$9"
  local concurrency="${10}"
  local seed="${11}"
  local temp="${12}"

  "${VLLM_BIN}" bench serve \
    --backend vllm \
    --model "${model}" \
    --served-model-name "${served}" \
    --dataset-name random \
    --random-input-len "${input_len}" \
    --random-output-len "${output_len}" \
    --random-range-ratio 0 \
    --num-prompts "${prompts}" \
    --max-concurrency "${concurrency}" \
    --temperature "${temp}" \
    --seed "${seed}" \
    --percentile-metrics ttft,tpot,itl,e2el \
    --metric-percentiles 50,90,95,99 \
    --host "${host}" \
    --port "${port}" \
    --save-result \
    --result-dir "${out_dir}" \
    --result-filename "${label}_bench.json" \
    | tee "${out_dir}/${label}_bench.log"
}

set +e
run_bench "a_long_prefill" "${RESULT_DIR}/a_long_prefill" "${A_HOST}" "${A_PORT}" "${A_MODEL}" "${A_SERVED_MODEL}" "${A_INPUT_LEN}" "${A_OUTPUT_LEN}" "${A_NUM_PROMPTS}" "${A_CONCURRENCY}" "${A_SEED}" "${A_TEMPERATURE}" &
A_PID=$!
run_bench "b_decode_sensitive" "${RESULT_DIR}/b_decode_sensitive" "${B_HOST}" "${B_PORT}" "${B_MODEL}" "${B_SERVED_MODEL}" "${B_INPUT_LEN}" "${B_OUTPUT_LEN}" "${B_NUM_PROMPTS}" "${B_CONCURRENCY}" "${B_SEED}" "${B_TEMPERATURE}" &
B_PID=$!

wait "${A_PID}"
A_STATUS=$?
wait "${B_PID}"
B_STATUS=$?
set -e

cleanup
trap - EXIT

if [[ "${A_STATUS}" -ne 0 || "${B_STATUS}" -ne 0 ]]; then
  echo "Mixed benchmark failed: A=${A_STATUS}, B=${B_STATUS}" >&2
  exit 1
fi

echo "Mixed benchmark completed successfully."
