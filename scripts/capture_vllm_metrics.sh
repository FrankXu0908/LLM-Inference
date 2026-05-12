#!/bin/bash
set -euo pipefail

# Capture vLLM Prometheus metrics once per interval.
#
# Example:
#   OUT=results/tables/Qwen3-8B/baseline_a_dp1_standard/c1_metrics.prom \
#   bash scripts/capture_vllm_metrics.sh

HOST="${HOST:-127.0.0.1}"
PORT="${PORT:-8000}"
INTERVAL="${INTERVAL:-1}"
OUT="${OUT:-metrics.log}"

mkdir -p "$(dirname "${OUT}")"

echo "# vLLM metrics capture"
echo "# endpoint=http://${HOST}:${PORT}/metrics"
echo "# interval=${INTERVAL}s"
echo "# output=${OUT}"

while true; do
  {
    echo "==== $(date -Is) ===="
    curl -s "http://${HOST}:${PORT}/metrics"
    echo
  } >> "${OUT}"
  sleep "${INTERVAL}"
done
