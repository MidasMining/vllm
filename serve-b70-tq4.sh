#\!/bin/bash
source /opt/intel/oneapi/setvars.sh --force 2>/dev/null
source ~/vllm-native/bin/activate
export VLLM_WORKER_MULTIPROC_METHOD=spawn
export ZE_AFFINITY_MASK=0
export VLLM_TARGET_DEVICE=xpu
export CCL_ZE_ENABLE=0
export FI_PROVIDER=tcp
export VLLM_XPU_ENABLE_XPU_GRAPH=1
vllm serve ~/llm-models/Qwen3.6-35B-A3B-int4-mixed-AutoRound \
  --served-model-name Qwen3.6-35B-A3B --dtype float16 \
  --port 8000 --host 0.0.0.0 --trust-remote-code \
  --gpu-memory-utilization 0.90 --max-model-len 16384 \
  --kv-cache-dtype turboquant_4bit_nc \
  --reasoning-parser qwen3 --max-num-seqs 128 \
  -cc.mode=0 -cc.cudagraph_mode=full
