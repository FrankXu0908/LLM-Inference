#!/bin/bash
set -euo pipefail

# Run a request-rate sweep with vLLM's native benchmark client.
#
# Example:
#   MODEL_CONFIG=configs/qwen3_8b_dense.yaml bash scripts/run_vllm_bench_request_rate.sh
#
# Override defaults:
#   RATES="1 2 4 8 12 16" NUM_PROMPTS=512 MAX_CONCURRENCY=32 bash scripts/run_vllm_bench_request_rate.sh

MODEL_CONFIG="${MODEL_CONFIG:-configs/qwen3_8b_dense.yaml}"
RATES="${RATES:-1 2 4 8 12 16}"
RESULT_DIR="${RESULT_DIR:-results/tables/Qwen3-8B/vllm_bench/request_rate}"

readarray -t CFG < <(python - "$MODEL_CONFIG" <<'PY'
import sys, yaml
cfg = yaml.safe_load(open(sys.argv[1], "r", encoding="utf-8")) or {}
m = cfg.get("model", {}) or {}
s = cfg.get("server", {}) or {}
sampling = cfg.get("sampling", {}) or {}
print(m.get("name", ""))
print(m.get("alias", "model"))
print(s.get("host", "127.0.0.1"))
print(s.get("port", 8000))
print(s.get("max_concurrent_requests", s.get("max_num_seqs", 32)))
print(sampling.get("temperature", 0.0))
print(sampling.get("max_tokens", 256))
PY
)

MODEL_NAME="${MODEL_NAME:-${CFG[0]}}"
MODEL_ALIAS="${MODEL_ALIAS:-${CFG[1]}}"
HOST="${HOST:-${CFG[2]}}"
PORT="${PORT:-${CFG[3]}}"
MAX_CONCURRENCY="${MAX_CONCURRENCY:-${CFG[4]}}"
TEMPERATURE="${TEMPERATURE:-${CFG[5]}}"
RANDOM_OUTPUT_LEN="${RANDOM_OUTPUT_LEN:-${CFG[6]}}"

RANDOM_INPUT_LEN="${RANDOM_INPUT_LEN:-512}"
NUM_PROMPTS="${NUM_PROMPTS:-256}"

if [[ -z "${MODEL_NAME}" ]]; then
  echo "model.name is missing in ${MODEL_CONFIG}"
  exit 1
fi

if [[ "${HOST}" == "0.0.0.0" ]]; then
  HOST="127.0.0.1"
fi

mkdir -p "${RESULT_DIR}"

echo "Model: ${MODEL_ALIAS} (${MODEL_NAME})"
echo "Endpoint: ${HOST}:${PORT}"
echo "Dataset: random, input=${RANDOM_INPUT_LEN}, output=${RANDOM_OUTPUT_LEN}, prompts=${NUM_PROMPTS}, max_concurrency=${MAX_CONCURRENCY}"
echo "Result dir: ${RESULT_DIR}"

for r in ${RATES}; do
  echo "===== request rate ${r} ====="
  vllm bench serve \
    --backend vllm \
    --model "${MODEL_NAME}" \
    --dataset-name random \
    --random-input-len "${RANDOM_INPUT_LEN}" \
    --random-output-len "${RANDOM_OUTPUT_LEN}" \
    --num-prompts "${NUM_PROMPTS}" \
    --max-concurrency "${MAX_CONCURRENCY}" \
    --request-rate "${r}" \
    --temperature "${TEMPERATURE}" \
    --host "${HOST}" \
    --port "${PORT}" \
    --save-result \
    --result-dir "${RESULT_DIR}" \
    --result-filename "request_rate_${r}.json"
done
