#!/usr/bin/env bash
set -euo pipefail

# Round-1: full sweep for one mode.
# Usage:
#   bash scripts/run_round1_sweep.sh tp2
#   bash scripts/run_round1_sweep.sh dp2
#   bash scripts/run_round1_sweep.sh dp2_ep
#
# Notes:
# 1) Start vLLM service with the matching mode before running this script.
# 2) After all three modes are done, comparison + figures are auto-generated.

MODE="${1:-}"
if [[ -z "${MODE}" ]]; then
  echo "Usage: bash scripts/run_round1_sweep.sh <tp2|dp2|dp2_ep>"
  exit 1
fi

if [[ "${MODE}" != "tp2" && "${MODE}" != "dp2" && "${MODE}" != "dp2_ep" ]]; then
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

echo "[round1] Running sweep for mode=${MODE}"
conda run -n "${CONDA_ENV}" python scripts/sweep_benchmark.py \
  --base-url "${BASE_URL}" \
  --model "${MODEL}" \
  --model-config "${MODEL_CONFIG}" \
  --parallel-mode "${MODE}"

TP2="results/tables/$(basename "${MODEL}")/tp2/results.json"
DP2="results/tables/$(basename "${MODEL}")/dp2/results.json"
DP2EP="results/tables/$(basename "${MODEL}")/dp2_ep/results.json"

if [[ -f "${TP2}" && -f "${DP2}" && -f "${DP2EP}" ]]; then
  echo "[round1] Found tp2/dp2/dp2_ep sweep results. Generating compare + figures..."
  conda run -n "${CONDA_ENV}" python scripts/sweep_compare_dp2_tp2.py
  conda run -n "${CONDA_ENV}" python scripts/plot_compare_tp2_dp2_dp2ep.py
  echo "[round1] Done."
else
  echo "[round1] Current mode sweep finished. Run other modes to enable compare plotting."
fi
