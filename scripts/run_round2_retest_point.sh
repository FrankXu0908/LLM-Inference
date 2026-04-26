#!/usr/bin/env bash
set -euo pipefail

# Round-2: anomaly point retest (default: c=16, input=256).
# Usage (run twice with different server modes):
#   bash scripts/run_round2_retest_point.sh dp2
#   bash scripts/run_round2_retest_point.sh dp2_ep
#
# After both modes are done, summary + round1-vs-round2 figure are auto-generated.

MODE="${1:-}"
if [[ -z "${MODE}" ]]; then
  echo "Usage: bash scripts/run_round2_retest_point.sh <dp2|dp2_ep>"
  exit 1
fi
if [[ "${MODE}" != "dp2" && "${MODE}" != "dp2_ep" ]]; then
  echo "Invalid mode: ${MODE}"
  exit 1
fi

CONDA_ENV="${CONDA_ENV:-vllm-dev}"
BASE_URL="${BASE_URL:-http://127.0.0.1:8000/v1}"
MODEL_CONFIG="${MODEL_CONFIG:-configs/model.yaml}"
MODEL="${MODEL:-$(python - "$MODEL_CONFIG" <<'PY'
import sys, yaml
cfg_path = sys.argv[1]
with open(cfg_path, "r", encoding="utf-8") as f:
    cfg = yaml.safe_load(f) or {}
print((cfg.get("model") or {}).get("name", ""))
PY
)}"
if [[ -z "${MODEL}" ]]; then
  echo "Failed to resolve model from ${MODEL_CONFIG}"
  exit 1
fi
REPEATS="${REPEATS:-5}"
CONCURRENCY="${CONCURRENCY:-16}"
INPUT_TOKENS="${INPUT_TOKENS:-256}"
MAX_TOKENS="${MAX_TOKENS:-128}"
NUM_REQUESTS="${NUM_REQUESTS:-32}"
TIMEOUT="${TIMEOUT:-600}"

ROOT="results/tables/$(basename "${MODEL}")/retest_c${CONCURRENCY}_in${INPUT_TOKENS}"
OUT_DIR="${ROOT}/${MODE}"
mkdir -p "${OUT_DIR}"

echo "[round2] mode=${MODE}, repeats=${REPEATS}, c=${CONCURRENCY}, input=${INPUT_TOKENS}"
for i in $(seq 1 "${REPEATS}"); do
  echo "[round2] ${MODE} run ${i}/${REPEATS}"
  conda run -n "${CONDA_ENV}" python scripts/benchmark_vllm.py \
    --base-url "${BASE_URL}" \
    --model "${MODEL}" \
    --model-config "${MODEL_CONFIG}" \
    --parallel-mode "${MODE}" \
    --prompt-type long \
    --input-tokens "${INPUT_TOKENS}" \
    --max-tokens "${MAX_TOKENS}" \
    --concurrency "${CONCURRENCY}" \
    --num-requests "${NUM_REQUESTS}" \
    --timeout "${TIMEOUT}" \
    --output-json "${OUT_DIR}/run${i}.json"
done

DP2_DIR="${ROOT}/dp2"
DP2EP_DIR="${ROOT}/dp2_ep"
if [[ -d "${DP2_DIR}" && -d "${DP2EP_DIR}" ]]; then
  dp2_count=$(ls -1 "${DP2_DIR}"/run*.json 2>/dev/null | wc -l || true)
  dp2ep_count=$(ls -1 "${DP2EP_DIR}"/run*.json 2>/dev/null | wc -l || true)
  if [[ "${dp2_count}" -ge 1 && "${dp2ep_count}" -ge 1 ]]; then
    echo "[round2] Found both modes. Generating summary + figure..."
    conda run -n "${CONDA_ENV}" python scripts/compare_retest_point.py \
      --root "${ROOT}" \
      --concurrency "${CONCURRENCY}" \
      --input-tokens "${INPUT_TOKENS}"
  fi
fi

echo "[round2] done."
