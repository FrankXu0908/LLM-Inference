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
print(cfg.get("attention_backend", nested.get("attention_backend", "")) or "")
print(cfg.get("flash_attn_version", nested.get("flash_attn_version", "")) or "")
print(cfg.get("tensor_parallel_size", nested.get("tensor_parallel_size", 1)))
print(cfg.get("data_parallel_size", nested.get("data_parallel_size", 1)))
print(cfg.get("max_model_len", nested.get("max_model_len", 10000)))
print(cfg.get("gpu_memory_utilization", nested.get("gpu_memory_utilization", 0.95)))
print(cfg.get("kv_cache_dtype", nested.get("kv_cache_dtype", "")) or "")
print(str(cfg.get("calculate_kv_scales", nested.get("calculate_kv_scales", False))).lower())
print(str(cfg.get("enforce_eager", nested.get("enforce_eager", False))).lower())
print(str(cfg.get("enable_prefix_caching", nested.get("enable_prefix_caching", True))).lower())
print(str(cfg.get("enable_chunked_prefill", nested.get("enable_chunked_prefill", True))).lower())
print(cfg.get("profiler", nested.get("profiler", "")) or "")
print(cfg.get("host", server.get("host", "0.0.0.0")))
print(cfg.get("port", server.get("port", 8000)))
print(cfg.get("max_num_seqs", server.get("max_num_seqs", 32)))
PY
)

MODEL_NAME="${CFG[0]}"
SERVED_MODEL_NAME="${CFG[1]}"
DTYPE="${CFG[2]}"
QUANTIZATION="${CFG[3]}"
ATTENTION_BACKEND="${CFG[4]}"
FLASH_ATTN_VERSION="${CFG[5]}"
TP_SIZE="${CFG[6]}"
DP_SIZE="${CFG[7]}"
MAX_LEN="${CFG[8]}"
GPU_UTIL="${CFG[9]}"
KV_CACHE_DTYPE="${CFG[10]}"
CALCULATE_KV_SCALES="${CFG[11]}"
ENFORCE_EAGER="${CFG[12]}"
ENABLE_PREFIX_CACHING="${CFG[13]}"
ENABLE_CHUNKED_PREFILL="${CFG[14]}"
PROFILER="${PROFILER:-${CFG[15]}}"
HOST="${HOST:-${CFG[16]}}"
PORT="${PORT:-${CFG[17]}}"
MAX_NUM_SEQS="${MAX_NUM_SEQS:-${CFG[18]}}"

if [[ -z "${MODEL_NAME}" ]]; then
  echo "model is missing in ${MODEL_CONFIG}"
  exit 1
fi

echo "Starting vLLM server with model: ${MODEL_NAME}"
echo "Configuration: served_model=${SERVED_MODEL_NAME:-<default>}, dtype=${DTYPE}, quantization=${QUANTIZATION:-none}, attention_backend=${ATTENTION_BACKEND:-auto}, flash_attn_version=${FLASH_ATTN_VERSION:-auto}, kv_cache_dtype=${KV_CACHE_DTYPE:-auto}, profiler=${PROFILER:-none}, tp=${TP_SIZE}, dp=${DP_SIZE}, max_len=${MAX_LEN}, max_num_seqs=${MAX_NUM_SEQS}, gpu_util=${GPU_UTIL}"

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
if [[ -n "${ATTENTION_BACKEND}" ]]; then
  CMD+=(--attention-backend "${ATTENTION_BACKEND}")
fi
if [[ -n "${FLASH_ATTN_VERSION}" ]]; then
  CMD+=(--flash-attn-version "${FLASH_ATTN_VERSION}")
fi
if [[ -n "${KV_CACHE_DTYPE}" ]]; then
  CMD+=(--kv-cache-dtype "${KV_CACHE_DTYPE}")
fi
if [[ "${CALCULATE_KV_SCALES}" == "true" ]]; then
  CMD+=(--calculate-kv-scales)
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
if [[ -n "${PROFILER}" ]]; then
  CMD+=(--profiler-config.profiler "${PROFILER}")
fi

exec "${CMD[@]}"
