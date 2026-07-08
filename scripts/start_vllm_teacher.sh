#!/usr/bin/env bash
set -euo pipefail

MODEL=${1:-${TEACHER_MODEL:-Qwen/Qwen2.5-7B-Instruct}}
PORT=${PORT:-8000}
API_KEY=${VLLM_API_KEY:-EMPTY}
DTYPE=${DTYPE:-auto}
TP=${TENSOR_PARALLEL_SIZE:-1}

# vLLM exposes an OpenAI-compatible HTTP API. Shadow-to-Think uses /v1/completions.
# Example:
#   TEACHER_MODEL=Qwen/Qwen2.5-7B-Instruct PORT=8000 bash scripts/start_vllm_teacher.sh
vllm serve "$MODEL" \
  --host 0.0.0.0 \
  --port "$PORT" \
  --dtype "$DTYPE" \
  --api-key "$API_KEY" \
  --tensor-parallel-size "$TP"
