#!/bin/bash
# GLM-5.3-Flash on 4x CMP 170HX — PP4 port of promisezackr/glm53-flash-170hx-pp8
# (their PP8 prod, halved). Tree: /home/user/vllm-glm53-487ecf187 (vLLM 487ecf187
# + their 24-patch set + local FlashMLA pin mirror).
# Differences vs their setup: AWQ-W4A16 compressed-tensors checkpoint (not
# NVFP4 ModelOpt); no DFlash2 drafter on disk -> MTP k=3 (their 51-71 t/s tier);
# 128k ctx first boot (4-card KV pool, not their 8-card 1M); port 8002 so DSv4
# prod keeps 8001. Partition 12,12,12,9 over 45 layers: the LAST rank also
# carries the unquantized MTP layer + lm_head (their PP8 ends ...,7,2 for the
# same reason) — 12,11,11,11 OOM'd GPU 3 at 61.4 GiB during profile. Sparse-MLA
# layers (3,7,...,43) per stage: 3/3/3/2.
V=${GLM_TREE:-/home/user/vllm-glm53-487ecf187}
cd "$V" || exit 1

_n=$(nvidia-smi -L 2>/dev/null | grep -c "^GPU ")
if [ "$_n" -ne 4 ]; then
  echo "PREFLIGHT FAIL: nvidia-smi sees $_n GPUs, expected 4." >&2
  exit 1
fi
export CUDA_HOME=/usr/local/cuda-13.3
export PATH=$CUDA_HOME/bin:$PATH
export CUDA_DEVICE_ORDER=PCI_BUS_ID
export CUDA_VISIBLE_DEVICES=${GLM_DEVS:-0,1,2,3}
export NCCL_P2P_DISABLE=1
export NCCL_IB_DISABLE=1
export VLLM_DO_NOT_TRACK=1
export HF_HUB_OFFLINE=1
export VLLM_WORKER_MULTIPROC_METHOD=spawn
export VLLM_PP_LAYER_PARTITION=12,12,12,9
export VLLM_PP_MAX_DECODE_REQS_PER_BATCH=2
export TRITON_CACHE_DIR=$V/triton-cache
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:False  # CMP VMM suspect: reserved=62.9GiB via expandable segments at the CE fault

# MTP spec decode: GLM_MTP=1 enables it. Default OFF pending the CE2 Xid 31
# REGION_VIOLATION during the MTP layer's weight load on PP rank 3 (twice
# reproducible at routed_experts _load_w13; layer 45 is the only unquantized
# MoE layer -> multi-GiB BF16 expert copy_, suspected oversized-CE-transfer
# trigger on CMP 170HX).
# With MTP on, rank 3 also carries the ~16GiB unquantized BF16 MTP block
# (measured: alloc 36.5->52.6GiB at layers.45.mtp_block) - shift 4 layers off
# the last stage so KV still fits at util 0.85.
SPEC_ARGS=""
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
  SPEC_ARGS='--speculative-config {"method":"dflash","model":"/mnt/ssd/models/GLM-5.3-Flash-DFlash2","num_speculative_tokens":7,"kv_cache_dtype":"auto"}'
elif [ "${GLM_MTP:-0}" = "1" ]; then
  export VLLM_PP_LAYER_PARTITION=13,13,13,6
  GLM_UTIL=0.90  # 0.85 cap was for the CE fault, now fixed at the allocator level
  SPEC_ARGS='--speculative-config {"method":"mtp","num_speculative_tokens":3}'
fi

$V/.venv/bin/python -m vllm.entrypoints.openai.api_server \
  --model /mnt/ssd/models/GLM-5.3-Flash-AWQ-W4A16 \
  --served-model-name glm53-flash \
  --pipeline-parallel-size 4 \
  --max-model-len ${GLM_LEN:-131072} \
  --max-num-seqs ${GLM_SEQS:-8} \
  --max-num-batched-tokens 4096 \
  --gpu-memory-utilization ${GLM_UTIL:-0.85} \
  --no-enable-prefix-caching \
  --disable-custom-all-reduce \
  $SPEC_ARGS ${GLM_KV_DTYPE:+--kv-cache-dtype $GLM_KV_DTYPE} \
  --enable-auto-tool-choice --tool-call-parser glm47 --reasoning-parser glm45 \
  --port 8002 --host 0.0.0.0
