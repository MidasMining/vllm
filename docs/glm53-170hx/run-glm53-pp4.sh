#!/bin/bash
# GLM-5.3-Flash on 4x CMP 170HX, vLLM pipeline parallel across the 4 cards.
# Production (2026-09-23): tree vllm-glm53-v0300 (branch glm53-pp4-v0300),
#   GLM_TREE=/home/user/vllm-glm53-v0300 GLM_DFLASH=1 GLM_LEN=1048576 \
#   GLM_KV_DTYPE=turboquant_k8v4 ./run-glm53-pp4.sh
# Benchmarked config uses GLM_LEN=262144. See docs/glm53-170hx/INSTALL.md.
# Env knobs: GLM_TREE, GLM_MODEL_DIR (default /mnt/ssd/models), GLM_PORT
# (8002), GLM_LEN, GLM_SEQS, GLM_UTIL, GLM_PART, GLM_DEVS, GLM_KV_DTYPE,
# GLM_DFLASH=1 (DFlash2 drafter) or GLM_MTP=1 (older MTP k=3 mode).
V=${GLM_TREE:-/home/user/vllm-glm53-v0300}
cd "$V" || exit 1

M=${GLM_MODEL_DIR:-/mnt/ssd/models}
PORT=${GLM_PORT:-8002}

_n=$(nvidia-smi -L 2>/dev/null | grep -c "^GPU ")
if [ "$_n" -ne 4 ]; then
  echo "PREFLIGHT FAIL: nvidia-smi sees $_n GPUs, expected 4." >&2
  exit 1
fi
# A previous server's workers can hold VRAM for minutes after SIGTERM while
# the driver scrubs it; overlapping CUDA processes can wedge CMP 170HX GSP.
if [ -n "$(nvidia-smi --query-compute-apps=pid --format=csv,noheader 2>/dev/null)" ]; then
  echo "PREFLIGHT FAIL: GPU compute processes still running; wait for them to exit." >&2
  exit 1
fi
if ss -ltn 2>/dev/null | grep -q ":$PORT "; then
  echo "PREFLIGHT FAIL: port $PORT is still in use." >&2
  exit 1
fi
export CUDA_HOME=${CUDA_HOME:-/usr/local/cuda-13.3}
export PATH="$CUDA_HOME/bin:$PATH"
export CUDA_DEVICE_ORDER=PCI_BUS_ID
export CUDA_VISIBLE_DEVICES=${GLM_DEVS:-0,1,2,3}
export NCCL_P2P_DISABLE=1
export NCCL_IB_DISABLE=1
export VLLM_DO_NOT_TRACK=1
export HF_HUB_OFFLINE=1
export VLLM_WORKER_MULTIPROC_METHOD=spawn
export VLLM_PP_LAYER_PARTITION=12,12,12,9
export VLLM_PP_MAX_DECODE_REQS_PER_BATCH=2
export TRITON_CACHE_DIR="$V/triton-cache"
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:False  # CMP VMM suspect: reserved=62.9GiB via expandable segments at the CE fault

# MTP spec decode: GLM_MTP=1 enables it. Default OFF pending the CE2 Xid 31
# REGION_VIOLATION during the MTP layer's weight load on PP rank 3 (twice
# reproducible at routed_experts _load_w13; layer 45 is the only unquantized
# MoE layer -> multi-GiB BF16 expert copy_, suspected oversized-CE-transfer
# trigger on CMP 170HX).
# With MTP on, rank 3 also carries the ~16GiB unquantized BF16 MTP block
# (measured: alloc 36.5->52.6GiB at layers.45.mtp_block) - shift 4 layers off
# the last stage so KV still fits at util 0.85.
SPEC_ARGS=()
# DFlash2 drafter (their prod config): GLM_DFLASH=1. Drafter is 2.3GB on the
# last rank (no 16GB MTP block), so the no-MTP partition fits; sidecar KV
# re-blocked to 256 per their patch 0024.
if [ "${GLM_DFLASH:-0}" = "1" ]; then
  export VLLM_PP_LAYER_PARTITION=${GLM_PART:-13,12,11,9}  # KV-ladder v0.30 2026-09-22: PCI01 caps rank0 at 13 MoE layers
  export VLLM_GLM5N_SIDECAR_BLOCK_SIZE=256
  GLM_UTIL=${GLM_UTIL:-0.93}
  # kv_cache_dtype:auto keeps the drafter's dense sliding-window/non-causal
  # layers on bf16 KV — no backend supports them under turboquant, and the
  # drafter cache is negligible next to the target. No-op when the main
  # cache is bf16.
  SPEC_ARGS=(--speculative-config "{\"method\":\"dflash\",\"model\":\"$M/GLM-5.3-Flash-DFlash2\",\"num_speculative_tokens\":7,\"kv_cache_dtype\":\"auto\"}")
elif [ "${GLM_MTP:-0}" = "1" ]; then
  export VLLM_PP_LAYER_PARTITION=13,13,13,6
  GLM_UTIL=0.90  # 0.85 cap was for the CE fault, now fixed at the allocator level
  SPEC_ARGS=(--speculative-config '{"method":"mtp","num_speculative_tokens":3}')
fi

KV_ARGS=()
if [ -n "${GLM_KV_DTYPE:-}" ]; then
  KV_ARGS=(--kv-cache-dtype "$GLM_KV_DTYPE")
fi

"$V/.venv/bin/python" -m vllm.entrypoints.openai.api_server \
  --model "$M/GLM-5.3-Flash-AWQ-W4A16" \
  --served-model-name glm53-flash \
  --pipeline-parallel-size 4 \
  --max-model-len "${GLM_LEN:-131072}" \
  --max-num-seqs "${GLM_SEQS:-8}" \
  --max-num-batched-tokens 4096 \
  --gpu-memory-utilization "${GLM_UTIL:-0.85}" \
  --no-enable-prefix-caching \
  --disable-custom-all-reduce \
  "${SPEC_ARGS[@]}" "${KV_ARGS[@]}" \
  --enable-auto-tool-choice --tool-call-parser glm47 --reasoning-parser glm45 \
  --port "$PORT" --host 0.0.0.0
