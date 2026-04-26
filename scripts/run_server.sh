#!/bin/bash
set -euo pipefail

# Start vLLM server from configs/model.yaml

MODEL_CONFIG="configs/model.yaml"

readarray -t CFG < <(python - "$MODEL_CONFIG" <<'PY'
import sys, yaml
cfg = yaml.safe_load(open(sys.argv[1], "r", encoding="utf-8")) or {}
m = cfg.get("model", {}) or {}
s = cfg.get("server", {}) or {}
print(m.get("name", ""))
print(m.get("dtype", "auto"))
print(m.get("tensor_parallel_size", 1))
print(m.get("data_parallel_size", 1))
print(m.get("max_model_len", 10000))
print(m.get("gpu_memory_utilization", 0.95))
print(str(m.get("enforce_eager", False)).lower())
print(str(m.get("enable_prefix_caching", True)).lower())
print(str(m.get("enable_chunked_prefill", True)).lower())
print(s.get("host", "0.0.0.0"))
print(s.get("port", 8000))
print(s.get("max_num_seqs", 32))
PY
)

MODEL_NAME="${CFG[0]}"
DTYPE="${CFG[1]}"
TP_SIZE="${CFG[2]}"
DP_SIZE="${CFG[3]}"
MAX_LEN="${CFG[4]}"
GPU_UTIL="${CFG[5]}"
ENFORCE_EAGER="${CFG[6]}"
ENABLE_PREFIX_CACHING="${CFG[7]}"
ENABLE_CHUNKED_PREFILL="${CFG[8]}"
HOST="${CFG[9]}"
PORT="${CFG[10]}"
MAX_NUM_SEQS="${CFG[11]}"

if [[ -z "${MODEL_NAME}" ]]; then
  echo "model.name is missing in ${MODEL_CONFIG}"
  exit 1
fi

echo "Starting vLLM server with model: ${MODEL_NAME}"
echo "Configuration: dtype=${DTYPE}, tp=${TP_SIZE}, dp=${DP_SIZE}, max_len=${MAX_LEN}, max_num_seqs=${MAX_NUM_SEQS}, gpu_util=${GPU_UTIL}"

CMD=(vllm serve "${MODEL_NAME}"
  --host "${HOST}"
  --port "${PORT}"
  --dtype "${DTYPE}"
  --tensor-parallel-size "${TP_SIZE}"
  --data-parallel-size "${DP_SIZE}"
  --max-model-len "${MAX_LEN}"
  --max-num-seqs "${MAX_NUM_SEQS}"
  --gpu-memory-utilization "${GPU_UTIL}"
)

if [[ "${ENFORCE_EAGER}" == "true" ]]; then
  CMD+=(--enforce-eager)
fi
if [[ "${ENABLE_PREFIX_CACHING}" == "true" ]]; then
  CMD+=(--enable-prefix-caching)
fi
if [[ "${ENABLE_CHUNKED_PREFILL}" == "true" ]]; then
  CMD+=(--enable-chunked-prefill)
fi

exec "${CMD[@]}"
