#!/usr/bin/env bash
# Serve base Qwen3-8B + the v4 and v5c LoRA adapters from ONE vLLM process
# (multi-LoRA), inside the katago-llm image, for the rank-evaluation bots.
#
#   bash src/rank_eval/serve_llm.sh start|stop|status|logs|wait
#
# Served model names (use these in llm_gtp.py --model / gtp2ogs configs):
#   qwen3-8b-base   un-fine-tuned Qwen3-8B
#   go-v4           runs/Qwen3-8B-GRPO-Go-Pro-v4/checkpoint-5000
#   go-v5c          runs/Qwen3-8B-GRPO-Go-Pro-v5c/checkpoint-5000
#
# GPU footprint is capped by GPU_MEM_UTIL (fraction of the card); the 8B
# weights alone need ~16.4 GB, so anything below ~0.20 will fail to start.
# `stop` removes the container and frees the GPU completely — that is the
# "pause" everybody else on the server needs (ogs_bots.sh calls it).
set -euo pipefail

CONTAINER=${CONTAINER:-ogs-llm}
IMAGE=${IMAGE:-katago-llm}
PORT=${PORT:-8100}
GPU_MEM_UTIL=${GPU_MEM_UTIL:-0.25}
MAX_MODEL_LEN=${MAX_MODEL_LEN:-1280}     # prompts <=518 tok + 512 completion (measured)
MAX_NUM_SEQS=${MAX_NUM_SEQS:-16}
WORKSPACE=${WORKSPACE:-/home/tyliu/katago_workspace}
HF_CACHE=${HF_CACHE:-/home/tyliu/hf_cache_docker}
FLASHINFER_CACHE=${FLASHINFER_CACHE:-/home/tyliu/flashinfer_cache_docker}
V4_CKPT=/workspace/runs/Qwen3-8B-GRPO-Go-Pro-v4/checkpoint-5000
V5C_CKPT=/workspace/runs/Qwen3-8B-GRPO-Go-Pro-v5c/checkpoint-5000

cmd=${1:-status}

start() {
  if docker ps --format '{{.Names}}' | grep -qx "$CONTAINER"; then
    echo "$CONTAINER already running"; return 0
  fi
  docker rm -f "$CONTAINER" >/dev/null 2>&1 || true
  docker run -d --name "$CONTAINER" --gpus all --network host --shm-size 8g \
    -v "$WORKSPACE:/workspace" -v "$HF_CACHE:/root/.cache/huggingface" \
    -v "$FLASHINFER_CACHE:/root/.cache/flashinfer" \
    -e HF_HUB_OFFLINE=1 -e VLLM_ATTENTION_BACKEND=FLASHINFER \
    -w /workspace "$IMAGE" \
    vllm serve Qwen/Qwen3-8B \
      --served-model-name qwen3-8b-base \
      --enable-lora --max-lora-rank 16 --max-loras 2 \
      --lora-modules "go-v4=$V4_CKPT" "go-v5c=$V5C_CKPT" \
      --dtype bfloat16 --max-model-len "$MAX_MODEL_LEN" --max-num-seqs "$MAX_NUM_SEQS" \
      --gpu-memory-utilization "$GPU_MEM_UTIL" --port "$PORT" --host 127.0.0.1
  echo "started $CONTAINER (port $PORT, gpu_mem_util $GPU_MEM_UTIL); 'serve_llm.sh wait' blocks until ready"
}

stop() {
  if docker ps -a --format '{{.Names}}' | grep -qx "$CONTAINER"; then
    docker rm -f "$CONTAINER" >/dev/null && echo "stopped $CONTAINER (GPU released)"
  else
    echo "$CONTAINER not running"
  fi
}

status() {
  if docker ps --format '{{.Names}}' | grep -qx "$CONTAINER"; then
    if curl -sf "http://127.0.0.1:$PORT/v1/models" >/dev/null 2>&1; then
      echo "$CONTAINER: RUNNING and ready on :$PORT"
      curl -s "http://127.0.0.1:$PORT/v1/models" | python3 -c 'import sys,json; print("  models:", ", ".join(m["id"] for m in json.load(sys.stdin)["data"]))'
    else
      echo "$CONTAINER: container up, server not ready yet (see 'logs')"
    fi
  else
    echo "$CONTAINER: not running"
  fi
  nvidia-smi --query-gpu=memory.used,memory.total --format=csv,noheader | sed 's/^/  GPU: /'
}

wait_ready() {
  local t=0
  until curl -sf "http://127.0.0.1:$PORT/v1/models" >/dev/null 2>&1; do
    if ! docker ps --format '{{.Names}}' | grep -qx "$CONTAINER"; then
      echo "container died; last log lines:"; docker logs --tail 40 "$CONTAINER" 2>&1; return 1
    fi
    sleep 5; t=$((t+5))
    if (( t % 60 == 0 )); then echo "  ...still loading (${t}s)"; fi
    if (( t > 1200 )); then echo "timeout waiting for server"; return 1; fi
  done
  echo "ready after ${t}s"; status
}

case "$cmd" in
  start) start ;;
  stop) stop ;;
  status) status ;;
  logs) docker logs --tail "${2:-100}" -f "$CONTAINER" ;;
  wait) wait_ready ;;
  *) echo "usage: $0 start|stop|status|logs|wait"; exit 1 ;;
esac
