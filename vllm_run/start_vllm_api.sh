#!/usr/bin/env bash
# Start vLLM API server; stdout/stderr go to ./vllm_deploy_logs/.
#
# Optional environment variables:
#   VLLM_MODEL        Model path (default: ./models/Qwen3-VL-8B-Instruct)
#   VLLM_API_MODEL    --served-model-name; defaults to VLLM_MODEL basename
#   VLLM_HOST         Listen address (default 0.0.0.0)
#   VLLM_PORT         Port (default 8001)
#   CUDA_VISIBLE_DEVICES  GPU id (default 0)

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
LOG_DIR="${SCRIPT_DIR}/vllm_deploy_logs"

MODEL_PATH="${VLLM_MODEL:-${SCRIPT_DIR}/models/Qwen3-VL-8B-Instruct}"
HOST="${VLLM_HOST:-0.0.0.0}"
PORT="${VLLM_PORT:-8001}"
SERVED_NAME=$(basename "${VLLM_API_MODEL:-$MODEL_PATH}")

mkdir -p "$LOG_DIR"
TS=$(date +%Y%m%d_%H%M%S)
LOG_FILE="${LOG_DIR}/vllm_${TS}.log"

export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
export VLLM_NO_VIDEO=1
export VLLM_SKIP_VISION_PROFILE=1
export VLLM_MODEL="$MODEL_PATH"

{
  echo "==== vLLM start $(date -Iseconds) ===="
  echo "MODEL_PATH=$MODEL_PATH"
  echo "SERVED_NAME=$SERVED_NAME"
  echo "LISTEN=${HOST}:${PORT}"
  echo "LOG_FILE=$LOG_FILE"
  echo "====================================="
} | tee -a "$LOG_FILE"

CUDA_VISIBLE_DEVICES="${CUDA_VISIBLE_DEVICES:-0}" vllm serve "$MODEL_PATH" \
  --port "$PORT" \
  --host "$HOST" \
  --tensor-parallel-size 1 \
  --dtype bfloat16 \
  --limit-mm-per-prompt '{"image": 100}' \
  --served-model-name "$SERVED_NAME" \
  --max-model-len 32768 \
  --max-num-batched-tokens 4096 \
  --max-num-seqs 16 \
  2>&1 | tee -a "$LOG_FILE"
