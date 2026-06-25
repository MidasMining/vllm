# vLLM on Intel Arc Pro B70 — b70-native branch

This branch contains patches and documentation for running vLLM on the
Intel Arc Pro B70 (BMG-G31, Xe2, 32GB GDDR6).

## Patches

### vllm (this repo)
- `inc_wna16_scheme.py` — Enable quantized MoE on XPU (removes platform guard)
- `xpu_worker.py` — CCL environment setup and single-GPU guard for B70
- `gpu_model_runner.py` — TurboQuant workspace pre-reservation for XPU Graph

### vllm-xpu-kernels (separate repo: MidasMining/vllm-xpu-kernels, branch b70-native)
- `mem_info.cpp` — Sysman API fallback for VRAM query on B70 Level Zero runtime

### Environment
- `libsycl.so.8 -> libsycl.so.9` symlink in oneAPI 2026.0 compiler directory

## Production Config

Model: Qwen3.6-35B-A3B AutoRound INT4 mixed (MoE, 35B/3B active)

```bash
VLLM_XPU_ENABLE_XPU_GRAPH=1 vllm serve <model_path> \
  --dtype float16 --gpu-memory-utilization 0.90 \
  --max-model-len 16384 --kv-cache-dtype turboquant_4bit_nc \
  --max-num-seqs 128 --block-size 32 \
  -cc.mode=0 -cc.cudagraph_mode=full
```

See `serve-b70-fp8.sh` and `serve-b70-tq4.sh` for complete serve scripts.

## Performance

- 72.9 tok/s single-user decode (c=1, FP8 KV)
- 300+ tok/s aggregate at c=8
- 615K KV cache tokens (TurboQuant 4bit_nc, 2.81x vs FP16)
- Beats V100 32GB by +39% on single-user decode at 14K context
- TQ4 matches or beats FP8 at 64K+ context with XPU Graph

## Key Findings

1. XPU Graph FULL mode: 6.5x speedup over eager (73 vs 11 tok/s)
2. TurboQuant 4bit_nc: 2.53x more KV tokens than FP8, <1% speed impact
3. TQ4 workspace fix enables 64K-131K context with XPU Graph
4. B70 exceeds V100 on decode by 39% with 3.5x better power efficiency

## Requirements

- Ubuntu 24.04+ with kernel 7.0.9+
- oneAPI 2026.0
- Intel Arc Pro B70 with Level Zero runtime 26.18.38308.1
- PyTorch 2.12.0 (source build against oneAPI 2026.0)
- vllm-xpu-kernels (source build from MidasMining/vllm-xpu-kernels b70-native)

See `b70-system-state.md` for complete setup details and benchmark results.
See `Pearl-B70-Architecture.md` for Pearl inference-mining integration analysis.

## Branch Contents

```
Patches:
  vllm/model_executor/layers/quantization/inc/schemes/inc_wna16_scheme.py
  vllm/v1/worker/gpu_model_runner.py
  vllm/v1/worker/xpu_worker.py

Documentation:
  B70-README.md                    (this file)
  b70-system-state.md              (hardware, software, benchmarks, findings)
  Pearl-B70-Architecture.md        (Pearl integration architecture)
  TurboQuant-Phase0-Task.md        (TQ SYCL Phase 0 task spec)
  TurboQuant-SYCL-Architecture.md  (TQ architecture analysis)

Serve scripts:
  serve-b70-fp8.sh                 (FP8 KV, 16K context)
  serve-b70-tq4.sh                 (TQ4 KV, 16K context)
  serve-b70-tq4-131k.sh            (TQ4 KV, 131K context)

Investigation (archived):
  b70-turboquant-sycl/             (Phases 0-2 standalone kernel code)
  b70-results/                     (matrix benchmarks, patch snapshots)
```
