#!/bin/bash
set -euo pipefail

# Start vLLM server from a YAML config.

MODEL_CONFIG="${MODEL_CONFIG:-configs/qwen3_8b_dense.yaml}"
VLLM_BIN="${VLLM_BIN:-vllm}"

readarray -t CFG < <(python - "$MODEL_CONFIG" <<'PY'
import sys, yaml
cfg = yaml.safe_load(open(sys.argv[1], "r", encoding="utf-8")) or {}
model = cfg.get("model", "")
if isinstance(model, dict):
    nested = model
    model = nested.get("name", "")
else:
    nested = {}
server = cfg.get("server", {}) or {}
print(model)
print(cfg.get("served_model_name") or nested.get("served_model_name") or "")
print(cfg.get("dtype", nested.get("dtype", "auto")))
print(cfg.get("quantization", nested.get("quantization", "")) or "")
print(cfg.get("tensor_parallel_size", nested.get("tensor_parallel_size", 1)))
print(cfg.get("data_parallel_size", nested.get("data_parallel_size", 1)))
print(cfg.get("max_model_len", nested.get("max_model_len", 10000)))
print(cfg.get("gpu_memory_utilization", nested.get("gpu_memory_utilization", 0.95)))
print(str(cfg.get("enforce_eager", nested.get("enforce_eager", False))).lower())
print(str(cfg.get("enable_prefix_caching", nested.get("enable_prefix_caching", True))).lower())
print(str(cfg.get("enable_chunked_prefill", nested.get("enable_chunked_prefill", True))).lower())
print(cfg.get("host", server.get("host", "0.0.0.0")))
print(cfg.get("port", server.get("port", 8000)))
print(cfg.get("max_num_seqs", server.get("max_num_seqs", 32)))
PY
)

MODEL_NAME="${CFG[0]}"
SERVED_MODEL_NAME="${CFG[1]}"
DTYPE="${CFG[2]}"
QUANTIZATION="${CFG[3]}"
TP_SIZE="${CFG[4]}"
DP_SIZE="${CFG[5]}"
MAX_LEN="${CFG[6]}"
GPU_UTIL="${CFG[7]}"
ENFORCE_EAGER="${CFG[8]}"
ENABLE_PREFIX_CACHING="${CFG[9]}"
ENABLE_CHUNKED_PREFILL="${CFG[10]}"
HOST="${CFG[11]}"
PORT="${CFG[12]}"
MAX_NUM_SEQS="${CFG[13]}"

if [[ -z "${MODEL_NAME}" ]]; then
  echo "model is missing in ${MODEL_CONFIG}"
  exit 1
fi

echo "Starting vLLM server with model: ${MODEL_NAME}"
echo "Configuration: served_model=${SERVED_MODEL_NAME:-<default>}, dtype=${DTYPE}, quantization=${QUANTIZATION:-none}, tp=${TP_SIZE}, dp=${DP_SIZE}, max_len=${MAX_LEN}, max_num_seqs=${MAX_NUM_SEQS}, gpu_util=${GPU_UTIL}"

CMD=("${VLLM_BIN}" serve "${MODEL_NAME}"
  --host "${HOST}"
  --port "${PORT}"
  --dtype "${DTYPE}"
  --tensor-parallel-size "${TP_SIZE}"
  --data-parallel-size "${DP_SIZE}"
  --max-model-len "${MAX_LEN}"
  --max-num-seqs "${MAX_NUM_SEQS}"
  --gpu-memory-utilization "${GPU_UTIL}"
)

if [[ -n "${SERVED_MODEL_NAME}" ]]; then
  CMD+=(--served-model-name "${SERVED_MODEL_NAME}")
fi
if [[ -n "${QUANTIZATION}" ]]; then
  CMD+=(--quantization "${QUANTIZATION}")
fi
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
