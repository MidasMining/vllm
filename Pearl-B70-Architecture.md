# Pearl Inference-Mining on B70 — Architecture Discussion

*Date: 2026-06-22*
*For: Claude ↔ GPT architectural discussion*
*Context: B70 vLLM serving is validated (73 tok/s, 615K KV with TQ4). The remaining question is Pearl inference-mining integration — which turns out to be a fundamentally different problem than "miner coexistence."*

---

## 1. What We Thought "Miner Coexistence" Meant

The v3 brief and Addendum D described miner coexistence as two separate programs sharing a GPU:

```
State A: miner alone
State B: vLLM loaded idle
State C: vLLM serving
State D: both running simultaneously
```

**This is wrong.** Pearl inference-mining is not two programs competing for GPU resources.

---

## 2. What Pearl Inference-Mining Actually Is

Pearl's `alpha-inference-miner` is a single integrated system where **inference matmuls ARE the mining proof-of-work.**

### Source references
- Docker image: https://hub.docker.com/r/jonathanquai/alpha-inference-miner
- vLLM plugin source: `MidasMining/pearl` → `miner/vllm-miner/src/vllm_miner/`
- CUDA GEMM kernels: `MidasMining/pearl` → `miner/pearl-gemm/csrc/`
- Live 5090 example: on the local network

### The state machine (from Docker readme, verified against source)

| State | What happens | Mining rate (5090) |
|-------|-------------|-------------------|
| Idle > 90s (model asleep) | Model offloaded from VRAM. Full-profile random 131072² GEMM mining. | ~385 TMAC/s |
| Between requests (model loaded) | Co-resident 16384² mining in leftover VRAM alongside loaded model. | ~240 TMAC/s |
| Chat prefill | **Merge-mining: the prompt matmuls produce both inference results AND mining shares from the same arithmetic.** | ~130 TMAC/s |
| Chat decode (default) | Mining paused. Pure inference. | 0 mining, ~50 tok/s |
| Chat decode (optional flag) | Mining during decode at reduced inference speed. | Mining active, ~34 tok/s |

### How merge-mining works

The model uses **int7 quantization** specifically because int7 GEMMs match Pearl's proof-of-work format. During a forward pass:

1. Input activations are quantized to int7 (symmetric, max_val=63)
2. Weights are stored as int7 (the `Qwen3.6-27B-heretic-pearl` model)
3. The GEMM kernel (`noisy_gemm`) performs `C = A @ B.T` while simultaneously:
   - Injecting cryptographic noise factors (EAL, EAR, EBL, EBR) into the computation
   - Computing BLAKE3 tensor hashes of intermediate results
   - Generating proof-of-work from the hash comparison against a target
4. The inference result (the correct matmul output) and the mining share (the proof-of-work hash) come from **the same GEMM operation**

From `gemm_operators.py`:
```python
def pearl_gemm_noisy(a, b, scale_a, scale_b, out_dtype, layer=None, submit_block=True):
    """Performs quantized matrix multiplication with cryptographic noise for blockchain mining.
    Computes C = A @ B.T while generating proof-of-work hashes from intermediate computations"""
```

### MoE-specific mining

For MoE models, `pearl_moe_method.py` implements `PearlMoEMethod`:
- Gate/up projections (GEMM1): int7 quantized, **mined** (noisy GEMM with proof-of-work)
- Down projections (GEMM2): FP8 block quantized, **NOT mined** (standard GEMM)
- Expert routing produces a canonical ordering (`MoERoutingLayout`) that is hashed as part of the proof

### VRAM management

From Jonathan (2026-06-18):
> "The model is even lazy loaded/offloaded from VRAM when there isn't a request for a while to maximize hashrate. The miner takes a few GB of scratch and gives it back when inference comes in."

The system dynamically manages VRAM between mining scratch space and model weights:
- Model offloads after 90s idle → full VRAM for mining
- Request arrives → model reloads, mining scratch shrinks
- `PEARL_VLLM_IDLE_MINE_VRAM_RESERVE_BYTES=134217728` (128 MiB floor)

---

## 3. What the Source Code Reveals About GPU Requirements

### Hard CUDA gates

From `vllm_kernels.py`:
```python
class PearlKernel(Int8ScaledMMLinearKernel):
    @classmethod
    def get_min_capability(cls) -> int:
        return 9  # SM90 = Hopper minimum

    @classmethod
    def can_implement(cls, c):
        if not current_platform.is_cuda():
            return False, "PearlKernel requires running on CUDA."
```

From `pearl_moe_method.py`:
```python
W2_WEIGHT_DTYPE = torch.float8_e4m3fn  # FP8 for down projection
```

### CUDA-specific kernel dependencies

The `pearl_gemm` package (imported throughout) provides:
- `gemm()` — standard int8 quantized GEMM
- `noisy_gemm()` — GEMM + noise injection + proof-of-work hash
- `noise_gen()` — GPU-side cryptographic noise generation
- `tensor_hash()` — GPU-side BLAKE3 hash
- `commitment_hash_from_merkle_roots()` — Merkle root computation
- `quantize()` — int7/int8 dynamic quantization
- `build_routing_data()` — MoE expert routing for proof

All of these are CUDA SM90+ kernels using CUTLASS 3.x (TMA, GMMA, warp specialization). None have SYCL/XPU equivalents.

### The quantization pipeline

From `quantization_operators.py`:
```python
MAX_VAL_7BIT = 63   # Mining mode: 7-bit symmetric
MAX_VAL_8BIT = 127  # Non-mining mode: 8-bit symmetric
```

Mining-mode int7 quantizes activations to [-63, +63] range. This specific range is tied to Pearl's proof-of-work protocol — the noise factors and hash verification assume int7 arithmetic.

---

## 4. What Exists on B70 That's Relevant

### From alphamine-intel (ESIMD miner investigation)

| Component | Status | Relevance |
|-----------|--------|-----------|
| ESIMD GEMM kernel (production) | 67.72 TMAC/s fused, 108 TMAC/s pure GEMM | Direct foundation for pearl_gemm SYCL equivalent |
| BLAKE3 test code (`blake3_test.cpp`) | Exists, status unclear | Foundation for tensor_hash SYCL equivalent |
| Int8 GEMM with dynamic quantization | Working | Close to int7 (same path, different max_val) |
| Noise generation (`gpu_noise.cpp`) | Exists | Foundation for noise_gen SYCL equivalent |
| GPU commit (`gpu_commit.cpp`) | Exists | Foundation for commitment hash SYCL equivalent |
| Tile geometry optimization | Extensive (50+ variants) | Directly applicable to pearl_gemm shapes |
| SLM cooperative patterns | Working | Needed for fused noise injection |

### From B70 vLLM work (this project)

| Component | Status | Relevance |
|-----------|--------|-----------|
| vLLM XPU Graph FULL serving | Production (73 tok/s) | The serving foundation Pearl would integrate with |
| MoE quantized serving (patched MoeWNA16Method) | Working | Proves MoE layers can run on XPU |
| FP8 KV cache | Working (388K tokens) | VRAM management for model weights + KV |
| TurboQuant KV | Working (615K tokens) | Additional KV capacity option |
| AutoRound INT4 MoE model loading | Working | Different quant format from Pearl's int7 |

---

## 5. The Gap Analysis — What B70 Needs for Pearl Inference-Mining

### Tier 1: Core GEMM + Proof-of-Work (must have)

| Component | CUDA version | SYCL equivalent needed | alphamine-intel starting point |
|-----------|-------------|----------------------|-------------------------------|
| `gemm()` — int8 quantized GEMM | CUTLASS 3.x SM90 | ESIMD DPAS GEMM | `xe_esimd_gemm*.cpp` (working) |
| `noisy_gemm()` — GEMM + noise + PoW | Fused CUTLASS + BLAKE3 | ESIMD fused GEMM + noise + hash | `b70_miner.cpp` (production, 67.72 TMAC/s) |
| `quantize()` — int7/int8 dynamic | CUDA kernel | SYCL kernel | Conceptually similar to existing quantize work |
| `tensor_hash()` — GPU BLAKE3 | CUDA kernel | SYCL kernel | `blake3_test.cpp` (exists) |
| `noise_gen()` — noise factors | CUDA kernel | SYCL kernel | `gpu_noise.cpp` (exists) |
| `commitment_hash_from_merkle_roots()` | CUDA kernel | SYCL kernel | Derivable from BLAKE3 |

**Critical question:** The alphamine-intel `b70_miner.cpp` already IS a fused GEMM + noise + BLAKE3 kernel for Pearl mining on B70. How much of it can be reused or adapted for the vllm-miner integration vs needing a new implementation?

### Tier 2: MoE Expert Mining (needed for MoE models)

| Component | CUDA version | SYCL equivalent needed |
|-----------|-------------|----------------------|
| `build_routing_data()` — expert routing for proof | CUDA kernel | SYCL kernel |
| `PearlMoEMethod` — vLLM MoE integration | Python + CUDA backends | Python + SYCL backends |
| Per-expert noisy GEMM with routing hash | CUDA kernel | SYCL kernel |

### Tier 3: vLLM Integration (Python-level)

| Component | Current state | What changes for XPU |
|-----------|--------------|---------------------|
| `PearlKernel.can_implement()` | Returns False for non-CUDA | Add `is_xpu()` path |
| `PearlKernel.get_min_capability()` | Returns 9 (SM90) | XPU has no SM concept — needs different capability check |
| `register_pearl_miner_layer()` | Initializes CUDA-based AsyncLoopManager | Need XPU-compatible async mining state |
| `mining_state.py` | Uses CUDA events (`enable_async_cuda_event_processing`) | Need XPU Level Zero events or equivalent |
| Model weights | `Qwen3.6-27B-heretic-pearl` int7 quantized | Same weights, different GEMM backend |
| VRAM management / lazy load | CUDA memory management | XPU memory management via Level Zero |

### Tier 4: Gateway / Pool Communication (probably unchanged)

| Component | Notes |
|-----------|-------|
| Pearl gateway (Unix domain socket) | Platform-independent, should work |
| MiningJob / block submission | Protocol-level, not GPU-specific |
| Pool communication | Network-level, not GPU-specific |

---

## 6. The Two B70 Paths for Pearl

### Path A: Dense int7 Pearl (matches current 5090 production)

Model: `Qwen3.6-27B-heretic-pearl` (int7 quantized, dense 27B)
Architecture: Same as the 5090 Docker setup — fused int7 GEMM + noise + BLAKE3 merge-mining.

**B70 dense performance context:**
- Current B70 dense vLLM (AutoRound INT4, XPU Graph FULL): 27.40 tok/s c=1
- The int7 Pearl model would be a different quant format (int7 vs INT4) with different GEMM kernels (pearl_gemm vs CUTLASS XE2)
- Dense model saturates at c=4 on B70 (all 27B params active per token)

**What's needed:**
- SYCL pearl_gemm equivalent (Tier 1 — alphamine-intel has the foundation)
- vLLM plugin XPU path (Tier 3 — Python-level, moderate effort)
- Int7 quantization SYCL kernel
- Lazy-load/offload on XPU

### Path B: MoE Pearl (better B70 economics, protocol question pending)

Model: Would need a MoE model quantized with Pearl's int7 format
Architecture: Same merge-mining concept but on MoE expert GEMMs

**B70 MoE performance context:**
- Current B70 MoE vLLM (AutoRound INT4, XPU Graph FULL): 72.91 tok/s c=1, 300 tok/s c=8
- MoE is dramatically better on B70 than dense
- But Pearl's current production model is dense, not MoE

**What's needed:**
- Everything from Path A, plus MoE-specific mining (Tier 2)
- A Pearl-compatible MoE model with int7 quantization
- Jonathan confirming MoE expert matmuls are valid proof-of-work
- Per-expert noisy GEMM with routing hash on SYCL

**The protocol question (for Jonathan):**
> Can Pearl inference-mining credit routed MoE expert matmuls? The current 5090 production setup mines on dense int7 GEMM. MoE only activates a subset of experts per token, producing fewer total GEMMs but faster inference. Does Pearl's proof/payment model support this, or does it require dense-layer coverage?

---

## 7. The alphamine-intel Connection

The `MidasMining/alphamine-intel` repo is not just "relevant" — it's the direct prototype for the Tier 1 kernel work. The production ESIMD miner (`b70_miner.cpp`) already does:

1. INT8 GEMM on XMX via ESIMD DPAS
2. Fused BLAKE3 hash computation
3. Noise generation and commitment
4. Per-tile proof-of-work checking
5. Runs at 67.72 TMAC/s on B70

The gap between alphamine-intel's standalone miner and Pearl's vllm-miner integration is:

| alphamine-intel (has) | vllm-miner (needs) |
|----------------------|-------------------|
| Standalone GEMM mining | GEMM integrated into vLLM forward pass |
| Random input matrices | Model weight matrices (A = activations, B = weights) |
| Fixed GEMM shapes | Variable shapes per layer/token count |
| No inference output needed | Must produce correct inference output AND mining proof |
| Standalone binary | vLLM plugin with Python-level integration |
| ESIMD kernel | SYCL kernel callable from PyTorch/vLLM |

The kernel math is the same (int8 GEMM + noise + BLAKE3). The integration surface is different.

---

## 8. Questions for Discussion

### Architecture questions

1. **Should B70 Pearl start with Path A (dense) or Path B (MoE)?**
   - Path A matches the existing 5090 production setup and avoids the MoE protocol question.
   - Path B has dramatically better B70 performance (73 vs 27 tok/s) but requires MoE proof-of-work validation.
   - Can we prototype Path A while waiting for Jonathan's answer on MoE?

2. **How much of alphamine-intel's b70_miner.cpp can be reused?**
   - The miner does standalone GEMM + noise + BLAKE3 on random matrices.
   - Pearl vllm-miner needs the same kernel but on model weight matrices, integrated into vLLM's forward pass.
   - Is this "adapt the existing kernel's entry point" or "rewrite with the same algorithm"?

3. **What's the right integration architecture?**
   - Option 1: Build a `pearl_gemm_sycl` package that mirrors `pearl_gemm`'s API but uses SYCL/ESIMD internally. The vllm-miner Python plugin calls `pearl_gemm_sycl` instead of `pearl_gemm`.
   - Option 2: Modify the vllm-miner plugin to dispatch to XPU kernels via PyTorch's `xpu` device. Less direct kernel control.
   - Option 3: Build the integration as a separate XPU-specific vllm-miner plugin that shares the protocol/state-machine code but has its own kernel dispatch.

4. **The int7 question for XPU:** Pearl's int7 (max_val=63) is a subset of int8 (max_val=127). On CUDA, the pearl_gemm `quantize()` kernel handles both. On XPU, the alphamine-intel ESIMD GEMM already does int8 DPAS. Can int7 be expressed as "int8 DPAS with clamped input range" or does it need a fundamentally different kernel? The DPAS instruction processes int8 operands — int7 values stored in int8 containers should work without kernel changes.

5. **VRAM management on XPU:** The lazy-load/offload pattern uses CUDA memory management. XPU would need Level Zero memory management for model weight offload/reload. Is this available in the current vLLM XPU stack, or does it need to be built?

### Scope questions

6. **Is this a weeks or months project?** The Tier 1 kernel work (pearl_gemm SYCL equivalent) is the core. alphamine-intel has much of the foundation. The vLLM integration (Tier 3) is Python-level plumbing. What's the realistic timeline?

7. **Can we test incrementally?** For example: first prove int7 GEMM on XPU produces correct output (correctness only), then add noise injection, then add BLAKE3 hashing, then add mining state management, then integrate with vLLM.

8. **What should CC investigate first?** The alphamine-intel repo has existing ESIMD GEMM + BLAKE3 code. CC could start by examining how close `b70_miner.cpp` is to what `pearl_gemm.noisy_gemm()` needs, producing a gap analysis rather than writing code.

---

## 9. Current B70 Production State (for reference)

```
Model: Qwen3.6-35B-A3B AutoRound INT4 mixed (MoE, 35B/3B active)
Backend: vLLM XPU Graph FULL
KV cache: turboquant_4bit_nc (615K tokens)
Decode: 72.91 tok/s c=1, 300 tok/s c=8
Quality: 20/22 Standard
Stability: 30 min verified, 0 errors
Patches: inc_wna16_scheme.py (MoE quant), xpu_worker.py (CCL), mem_info.cpp (Sysman)
```

This is a working inference server. Pearl integration would add mining proof-of-work to the forward pass, not replace the serving capability.

---

*This document is for architectural discussion between Claude and GPT. No code should be written until the approach is agreed and Jonathan's MoE question is answered.*
