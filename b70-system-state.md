# Intel Arc Pro B70 — System State

*Last updated: 2026-06-22*

## Hardware

| Component | Value |
|-----------|-------|
| GPU | Intel Arc Pro B70 (BMG-G31, device 0xe223) |
| VRAM | 32 GB GDDR6 |
| CPU | AMD EPYC 7R32 48-Core (96 threads) |
| RAM | 126 GB |
| Storage | 233 GB NVMe + 256 GB USB flash (/mnt/flash) |
| Network | 192.168.1.57 (SSH: user/1) |
| Render Node | /dev/dri/card1 |
| GPU Count | 1 |

## System Software

| Component | Version |
|-----------|---------|
| OS | Ubuntu 24.04.4 LTS |
| Kernel | 7.0.9-070009-generic (Xe2 Graphics driver built-in) |
| xe driver | Intel Xe2 Graphics module |
| Level Zero loader | 1.28.2 |
| Level Zero GPU runtime | 26.18.38308.1 |
| libze-dev | 1.21.9.0 |
| Docker | Installed, active (no containers/images) |
| Python | 3.12.3 (system) |

## oneAPI Stack

| Component | Version | Path |
|-----------|---------|------|
| DPC++/C++ Compiler | 2026.0.0.20260331 | /opt/intel/oneapi/compiler/2026.0 |
| oneAPI 2025.3 | Also installed (pip torch runtime deps) | /opt/intel/oneapi/compiler/2025.3 |
| oneCCL (BMG) | 2021.15.9.14 | /opt/intel/oneapi/ccl/2021.15 |
| MKL, TBB, MPI, PTI | Latest via oneAPI 2026.0 | /opt/intel/oneapi/ |
| xpu-smi | Installed, working | System path |

## Models on Disk

### NVMe (~89 GB free)

| Model | Path | Size | Working? |
|-------|------|------|----------|
| Intel/Qwen3.6-35B-A3B-int4-mixed-AutoRound | ~/llm-models/Qwen3.6-35B-A3B-int4-mixed-AutoRound | 21 GB | YES (vLLM patched + XPU Graph) |
| bartowski/Qwen3.6-35B-A3B-Q4_K_M.gguf | ~/llm-models/gguf/ | 21 GB | YES (llama.cpp SYCL) |

### Flash Drive (/mnt/flash, ~188 GB free)

| Model | Path | Size | Working? |
|-------|------|------|----------|
| Intel/Qwen3.6-27B-int4-AutoRound | /mnt/flash/llm-models/Qwen3.6-27B-int4-AutoRound | 18 GB | YES (vLLM native) |
| Qwen/Qwen3-14B | /mnt/flash/llm-models/Qwen3-14B | 28 GB | YES (INT4 only on llm-scaler) |

Previously downloaded and deleted: Qwen3.6-27B (52 GB, garbage output), Qwen3.6-27B-heretic (61 GB), Qwen3-14B-heretic (56 GB), Qwen3.6-27B-AWQ-INT4 (20 GB, zero points unsupported), Qwen3.6-27B-FP8 (29 GB, OOM).

## Installed Inference Backends

### 1. Native vLLM (upstream, from source) — RUNNING (MoE, patched)

| Component | Version | Source |
|-----------|---------|--------|
| vLLM | 0.1.dev1+gd272418f4 | ~/vllm-src (editable) |
| PyTorch | 2.12.0a0+git0d62256 | ~/pytorch-src (source build, oneAPI 2026.0) |
| vllm-xpu-kernels | 0.1.11.dev0+g11f42aa | ~/vllm-xpu-kernels-src (source build) |
| auto-round-lib | 0.14.0.dev202606202218 | ~/auto-round-src (source build) |
| torchvision | 0.27.0+78839c2 | ~/torchvision-src (source build) |
| triton-xpu | 3.7.1 | pip |
| Python venv | ~/vllm-native | Python 3.12 |

**Serve scripts:** `~/serve-native.sh` (dense 27B), `~/serve-moe.sh` (MoE 35B-A3B eager), `~/serve-moe-xpugraph-full.sh` (MoE 35B-A3B XPU Graph)

**Current server:** MoE 35B-A3B on port 8000, 16K context, FP8 KV cache, XPU Graph FULL mode, 0.90 gpu-mem-util (389K KV tokens)

**Key env vars:** `VLLM_WORKER_MULTIPROC_METHOD=spawn ZE_AFFINITY_MASK=0 VLLM_TARGET_DEVICE=xpu CCL_ZE_ENABLE=0 FI_PROVIDER=tcp VLLM_XPU_ENABLE_XPU_GRAPH=1`

**Patches applied:**
- `vllm/v1/worker/xpu_worker.py` — module-level CCL env vars, single-GPU all_reduce guard
- `vllm-xpu-kernels/csrc/utils/mem_info.cpp` — replaced unavailable L0 ext API with Sysman API
- `libsycl.so.8 -> libsycl.so.9` symlink in oneAPI 2026.0 compiler dir
- `inc/schemes/inc_wna16_scheme.py` — removed is_xpu() from MoE guard, enables quantized MoE on XPU (debug logging removed, clean commit)
- Patch snapshots saved at `~/b70-vllm/snapshots/pre-moe-*`
- **Both patches committed to `MidasMining/vllm` branch `b70-native`** (commit 64ece2d, pushed 2026-06-22)

**Status:**
- Dense models: WORKING (Qwen3.6-27B AutoRound INT4)
- MoE models: WORKING (Qwen3.6-35B-A3B AutoRound INT4, patched guard)
- MoE method: AutoGPTQMoEMethod -> WNA16 XPU backend -> CUTLASS grouped GEMM
- XMX INT8 compute: ACTIVE (oneAPI >= 2026 detected, no FP16 fallback)
- Flash Attention v2: YES
- XPU Graph FULL mode: **YES** (mode=0 + cudagraph_mode=full, 6.5x speedup)
- torch.compile: NO (inductor has no XPU backend, piecewise cudagraph fails)
- XCCL distributed: NO (falls back to gloo, not needed for single GPU)

### 2. llama.cpp SYCL — INSTALLED

| Component | Version | Path |
|-----------|---------|------|
| llama.cpp | commit 8a118ee86 (build 9746) | ~/llama.cpp |
| Build flags | `-DGGML_SYCL=ON -DCMAKE_C_COMPILER=icx -DCMAKE_CXX_COMPILER=icpx` | ~/llama.cpp/build |

**Binaries:** `~/llama.cpp/build/bin/` (llama-bench, llama-cli, llama-server)

**Status:** Fully functional on BMG. 57+ tok/s decode on MoE 35B Q4_K_M.

### 3. LLM-Scaler (Intel vLLM fork) — NOT INSTALLED

Docker containers and images were removed to free disk space. Can be restored with:
```bash
docker pull intel/llm-scaler-vllm:0.14.0-b8.3.1
```

Was the fastest vLLM backend tested (16.36 tok/s c=1 for 27B AutoRound).
Does NOT support BMG architecture for MoE models (sycl_arch not recognized).

### 4. OpenVINO — NOT INSTALLED

### 5. SGLang — NOT INSTALLED

XPU backend is experimental.

### 6. Ollama — NOT INSTALLED

Upstream SYCL PR (#11160) not yet merged.

## Benchmark Results Summary

### Qwen3.6-35B-A3B MoE — All Backends

| Model | Backend | c=1 tok/s | c=4 tok/s | c=8 tok/s | TPOT c=1 | KV Cache | VRAM |
|-------|---------|----------|----------|----------|----------|----------|------|
| MoE AutoRound INT4 | **Patched vLLM + XPU Graph** | **72.91** | **173.26** | **300.16** | **13.71ms** | 388,747 | 25.8 GB |
| MoE Q4_K_M | llama.cpp SYCL | 57.16 | -- | -- | 17.5ms | -- | 21.5 GB |
| MoE AutoRound INT4 | Patched vLLM (eager) | 11.18 | 43.32 | 86.36 | 89.23ms | 218,949 | 25.8 GB |

XPU Graph FULL mode eliminates Python/framework overhead. vLLM+XPU Graph now exceeds llama.cpp hardware ceiling.

### Standardized llm-bench Results (v3.0, 2026-06-22)

**Run:** `b70-moe-xpugraph` — 16K context, 0.90 gpu-mem-util, FP8 KV (389K tokens)
**Results:** `~/llm-bench/results/b70-moe-xpugraph-v3.0-20260621-235506/`

#### Quality: 20/22 (90.9%)

| Test | Score | TTFT | Tokens |
|------|-------|------|--------|
| ZMQ Listener Bug | 2/2 PASS | 0.18s | 6,712 |
| PPLNS Mining Pool Bugs | 3/3 PASS | 0.19s | 7,270 |
| Payment System Bugs | **2/4 FAIL** | 0.16s | 7,219 |
| Stratum Protocol Bugs | 5/5 PASS | 0.24s | 7,892 |
| HiveOS Wrapper Creation | 8/8 PASS | 0.18s | 6,495 |

Missed checks: SQL injection and atomicity in Payment System.
Throughput: 82.0 t/s avg. TTFT: 0.19s avg.
Thinking was disabled for quality tests (enable_thinking=false).

#### Decode Rate: 70.8 t/s median (stable across context)

| Context | TTFT | Decode Rate |
|---------|------|-------------|
| 500 | 0.115s | 71.6 t/s |
| 2,000 | 0.294s | 72.2 t/s |
| 4,000 | 0.513s | 71.5 t/s |
| 8,000 | 1.078s | 69.8 t/s |
| **14,000** | 2.038s | **68.6 t/s** |

Decode rate extremely consistent — only 4% drop from 500 to 14K context. TTFT scales linearly.

#### Context Scaling: Stable to 64K (Phase 4)

| Context | TTFT | Decode Rate | max-model-len |
|---------|------|-------------|---------------|
| 500 | 0.15s | 71.7 t/s | 64K |
| 14,000 | 2.02s | 68.6 t/s | 64K |
| 30,000 | 5.26s | 64.8 t/s | 64K |
| 48,000 | 10.13s | 61.1 t/s | 64K |
| **60,000** | 14.23s | **58.9 t/s** | 64K |

Only 18% degradation from 500 to 60K context. Still above V100 target (49.4 t/s) at 60K.
32K context: 450,128 KV tokens (13.74x concurrency). 64K context: 488,711 KV tokens (7.46x concurrency).

#### Parallel Throughput: Peak 820.3 t/s @ c=64

**Initial sweep** (run_all.py, `b70-moe-xpugraph`):

| c | Throughput | Per-Req | TTFT | p95 Lat |
|---|-----------|---------|------|---------|
| 1 | 60.3 t/s | 60.3 | 0.11s | 0.5s |
| 2 | 75.0 t/s | 37.5 | 0.21s | 2.1s |
| 4 | 99.0 t/s | 24.8 | 0.29s | 14.5s |
| 8 | 167.0 t/s | 20.9 | 0.23s | 15.7s |
| 16 | 297.9 t/s | 18.6 | 0.29s | 21.4s |

**Extended sweep** (`b70-moe-parallel-sweep`):

| c | Throughput | Per-Req | TTFT | p95 Lat |
|---|-----------|---------|------|---------|
| 1 | 59.5 t/s | 59.5 | 0.12s | 0.5s |
| 8 | 166.8 t/s | 20.9 | 0.23s | 15.7s |
| 16 | 296.7 t/s | 18.5 | 0.29s | 21.5s |
| 24 | 470.8 t/s | 19.6 | 0.36s | 20.9s |
| 32 | 494.5 t/s | 15.5 | 0.42s | 25.2s |
| 48 | 747.6 t/s | 15.6 | 0.53s | 26.4s |
| **64** | **820.3 t/s** | 12.8 | 0.64s | 31.8s |

11.5x speedup at peak. Still climbing at c=64. Auto-detect probing hit 1052 t/s.
Sweet spot: c=8–16 for interactive, c=48–64 for batch throughput.

#### Long Context: 3/3 PASS @ 16K

| Test | Result |
|------|--------|
| Needle-in-haystack | PASS |
| Multi-file understanding | PASS (100%) |
| Cross-reference retrieval | PASS (4/4) |

Initially failed due to null content from reasoning model spending all tokens on `<think>` tags. Fixed by disabling thinking for comprehension tests.

#### BWA-MEM2 Domain Test: Response Captured, Scoring Pending

- Thinking enabled (15K tokens): model spent ALL tokens thinking, no content produced
- Thinking disabled (8K tokens): initial analysis produced but hit severe repetition loop at ~1500 tokens
- Raw response saved to `~/llm-bench/results/b70/bwamem2_response.txt`
- Scoring deferred to Midas (manual 30-point rubric)
- V100 baseline: 18/30 (60%)

#### Stability: 30 min @ c=4, 0 Errors

| Metric | Value |
|--------|-------|
| Duration | 30.06 minutes |
| Total requests | 1,224 |
| Successful | 1,224 (100%) |
| Errors | 0 |
| tok/s first 10 batches | 48.6 |
| tok/s last 10 batches | 48.9 |
| Drift | +0.5% (warmup only) |

Server maintained rock-solid 48.9 tok/s per request at c=4 for 30 minutes.

#### FP8 KV Cache: Verified Working (Phase 4)

| Metric | FP8 KV (fp8_e5m2) | Default KV (auto/FP16) |
|--------|-------------------|----------------------|
| Available KV memory | 5.28 GiB | 5.28 GiB |
| KV tokens | **388,747** | **219,028** |
| Max concurrency @16K | 23.73x | 13.37x |
| c=1 tok/s | 72.9 | 72.1 |

**Compression ratio: 1.77x.** FP8 provides 77% more KV tokens with < 1% speed impact.

#### Concurrency × Context Matrix (FP8 KV Baseline)

Maps the practical operating envelope: what happens when concurrency and context depth are pushed simultaneously. This is the "before" ruler for TurboQuant comparison.

**Per-request decode tok/s:**

| | 4K ctx | 16K ctx | 32K ctx | 64K ctx | 128K ctx |
|-----|--------|---------|---------|---------|----------|
| c=1 | 71.5 | 68.6 | 64.6* | 66.3 | 52.7 |
| c=2 | 51.2 | 46.2 | 36.0 | 31.0 | 24.2 |
| c=4 | 43.1 | 32.8 | 22.3 | 17.3 | -- |
| c=8 | 34.7 | 20.9 | 14.5 | -- | -- |
| c=16 | 25.8 | 13.4 | -- | -- | -- |

**Aggregate tok/s:**

| | 4K ctx | 16K ctx | 32K ctx | 64K ctx | 128K ctx |
|-----|--------|---------|---------|---------|----------|
| c=1 | 71.5 | 68.6 | 64.6* | 66.3 | 52.7 |
| c=2 | 89.0 | 57.6 | 23.2 | 15.0 | 4.8 |
| c=4 | 142.6 | 68.2 | 26.5 | 14.8 | -- |
| c=8 | 210.7 | 80.6 | 34.4 | -- | -- |
| c=16 | 281.8 | 85.9 | -- | -- | -- |

**TTFT (seconds):**

| | 4K ctx | 16K ctx | 32K ctx | 64K ctx | 128K ctx |
|-----|--------|---------|---------|---------|----------|
| c=1 | 0.5* | 2.0* | 5.3* | 15.7 | 48.3 |
| c=2 | 0.80 | 3.56 | 8.75 | 23.9 | 73.9 |
| c=4 | 1.26 | 5.97 | 14.6 | 40.3 | -- |
| c=8 | 2.27 | 10.8 | 26.5 | -- | -- |
| c=16 | 4.32 | 20.6 | -- | -- | -- |

\* = prior benchmark data (different measurement script, consistent methodology)
-- = exceeds FP8 KV budget (388K tokens)

**Key findings:**
- Decode speed degrades ~2x from c=1→c=8 at any context, and ~1.4x from 4K→128K at any concurrency
- At c=2@64K (real-world deep-context pair), per-request decode is still 31 tok/s — usable
- At c=4@32K (interactive multi-user), per-request decode is 22 tok/s with 14.6s TTFT — borderline
- At c=8@16K (standard multi-user), per-request decode is 20.9 tok/s with 10.8s TTFT — practical limit
- TTFT scales roughly linearly with both context and concurrency (prefill is compute-bound)
- Aggregate tok/s peaks at high concurrency + short context (281.8 at c=16@4K) but long-context destroys aggregate throughput

**Sweet spot analysis:**
1. **Best single-user deep context:** c=1@128K — 52.7 tok/s, 48.3s TTFT (usable if TTFT is acceptable)
2. **Best multi-user balanced:** c=2@32K — 36 tok/s per-req, 8.75s TTFT
3. **Best throughput-optimized:** c=8@4K — 210.7 agg tok/s, 2.27s avg TTFT
4. **Pearl serving recommendation:** c=4@16K — 32.8 tok/s per-req with 5.97s TTFT, 68.2 agg tok/s, fits 4 concurrent 16K sessions in 388K KV budget

**TurboQuant impact projection (FP8+TQ4, ~1.47M KV tokens):**
- c=4@64K would become possible (currently exceeds budget)
- c=8@32K would have 4.6x headroom instead of being at 66% capacity
- c=2@128K with room for 5 more concurrent 128K sessions
- The entire matrix shifts right by ~3.8x on the context axis

**Results:** `~/b70-vllm/results/matrix/` and benchmark script `~/matrix_bench.py`

### TurboQuant Investigation (Phase 4)

Built `TheTom/llama-cpp-turboquant` fork with Vulkan backend on B70.

| Backend | KV Type | Decode t/s | PP512 t/s | Notes |
|---------|---------|-----------|----------|-------|
| SYCL | default | 57.16 | 992.74 | Existing llama.cpp build |
| Vulkan | default | 36.10 | 872.03 | ~37% slower than SYCL |
| Vulkan | turbo4 | 35.25 | 828.24 | **-2.4% vs Vulkan default** |
| Vulkan | turbo3 | 35.60 | 841.02 | **-1.4% vs Vulkan default** |

TurboQuant works on B70 Vulkan with negligible overhead. MTP not available in this fork.

**SYCL backend has zero TurboQuant support.** Porting scope: moderate (2-3 days). Key items:
- WHT butterfly kernels (128-element shared memory, 7 butterfly stages)
- Centroid lookup + bit-packing dequant kernels
- Flash attention integration (template-based tile/vec in SYCL vs spec-constant in Vulkan)
- `GGML_OP_TURBO_WHT` kernel dispatch (graph already emits these ops)

Implementation uses Randomized Hadamard Transform (RHT) with Lloyd-Max optimal centroids for N(0,1).
Turbo4: 4-bit, 128 elements/block, 16 centroids, ~3.8x KV compression.
Turbo3: 3-bit, 128 elements/block, 8 centroids, ~4.9x KV compression.

### TurboQuant SYCL Phase 0 — Standalone Microbenchmark (Phase 0)

Standalone SYCL correctness and microbenchmark harness for TurboQuant dequantization on B70.
**Project:** `~/turboquant-sycl/` on rig, `/tmp/turboquant-sycl/` locally.
**Architecture docs:** `~/projects/pearl/vllm/TurboQuant-SYCL-Architecture.md`, `TurboQuant-Phase0-Task.md`.

#### Correctness: 7/7 PASS

| Test | Result | Detail |
|------|--------|--------|
| WHT round-trip | PASS | Max error: 2.38e-7 |
| Turbo3 pack/unpack | PASS | Bit-exact |
| Turbo4 pack/unpack | PASS | Bit-exact |
| Turbo3 quant/dequant (10K vecs) | PASS | NMSE: 0.0342 (paper: ~0.034) |
| Turbo4 quant/dequant (10K vecs) | PASS | NMSE: 0.0184 (paper: ~0.009-0.019) |
| SYCL GPU Turbo3 vs CPU ref | PASS | Max diff: 0.0 (exact) |
| SYCL GPU Turbo4 vs CPU ref | PASS | Max diff: 0.0 (exact) |

#### Benchmark: 6 Modes, 5 Token Scales (Qwen3.6: d_head=128, n_heads=40)

Steady-state ns/vector at scale (128K tokens):

| Mode | Description | Turbo4 ns/vec | Turbo3 ns/vec |
|------|-------------|---------------|---------------|
| 1. Unpack only | Bit manipulation | 1.12 | 1.11 |
| 2. Unpack + codebook | + table lookup | 1.23 | 1.29 |
| 3. Unpack + CB + WHT | Full dequant, no write | 2.93 | 3.31 |
| 4. Full dequant + write | **Split approach cost** | **2.79** | **3.20** |
| 5. Full dequant + checksum | ALU-only, no write | 3.76 | 4.07 |
| 6. Pack (quantize) | Write path | ~6.2 (1K) | ~6.9 (1K) |

#### Critical Finding: Mode 4 vs Mode 5

**Mode 4 (global write) is 21–26% FASTER than Mode 5 (checksum reduction).**

| Scale | Turbo4 M4 | Turbo4 M5 | M4 faster by |
|-------|-----------|-----------|-------------|
| 1K tokens | 332 us | 370 us | 10.5% |
| 4K tokens | 954 us | 1144 us | 16.6% |
| 14K tokens | 3135 us | 4223 us | 25.8% |
| 60K tokens | 13414 us | 18077 us | 25.8% |
| 128K tokens | 28611 us | 38549 us | 25.8% |

The global FP32 write is free. The barrier-heavy tree reduction in Mode 5 is more expensive than coalesced global store in Mode 4. Dequant cost is entirely ALU-bound (WHT butterfly stages + barriers).

#### Estimated ms/token for Dequant Only (Mode 4, turbo4)

| Context | ms/token |
|---------|----------|
| 14K | 3.13 |
| 60K | 13.4 |
| 128K | 28.6 |

#### Phase 0 Decision

Per the decision matrix: Mode 4 at 14K = 3.13 ms > 1.0 ms threshold → **Skip Phase 1 (split), proceed directly to Phase 2 (fused kernel).**

Fusing with attention will eliminate the global write entirely and amortize WHT cost across the dot product, which is the dominant cost center.

#### Effective Bandwidth

| Metric | Value |
|--------|-------|
| Packed read bandwidth | 24.3 GB/s (turbo4) |
| FP32 write bandwidth | 183 GB/s |
| Peak unpack bandwidth | 60.8 GB/s |
| Device global memory | 32,656 MB |

#### Mode 6 (Pack) Note

Pack kernel works at 1K/4K tokens (~6-7 ns/vector, 2.2x slower than dequant). At 14K+ tokens, hits GPU TDR timeout (DEVICE_LOST) — needs kernel splitting or reduced batch sizes for large KV caches.

### TurboQuant SYCL Phase 0.5 — Rotated-Space Attention (Phase 0.5)

Measures the speedup from computing attention in rotated (Hadamard) space, which eliminates WHT from the per-token inner loop. Since WHT is orthogonal: `q·k = (Hq)·(Hk)` and `output = H⁻¹(Σ wᵢ·z_vᵢ)`.

**Correctness: 3/3 PASS**

| Test | Result | Detail |
|------|--------|--------|
| CPU standard vs rotated attention | PASS | Max rel error: 3.9e-7 |
| GPU K-scores vs CPU reference | PASS | Max abs error: 5.7e-6 |
| GPU Path D vs CPU standard attention | PASS | Max rel error: 6.9e-7 |

**Benchmark: Path A (baseline dequant) vs Path D (rotated-space attention)**

| n_tokens | Path A (us) | Path D (us) | Speedup | Path D est ms/tok @14K |
|----------|------------|------------|---------|----------------------|
| 1K | 333 | 123 | **2.70x** | 1.72 |
| 4K | 921 | 420 | **2.19x** | 1.47 |
| 14K | 3,037 | 1,607 | **1.89x** | 1.61 |
| 60K | 12,995 | 6,737 | **1.93x** | 1.57 |
| 128K | 27,711 | 14,325 | **1.93x** | 1.57 |

**Path breakdown at steady state (128K):**

| Component | Time (us) | ns/vec | % of Path D |
|-----------|----------|--------|-------------|
| Path B (K-score: unpack+codebook+dot) | 12,020 | 2.35 | 84% |
| Path C (V-accum: unpack+codebook+acc) | 2,143 | 0.42 | 15% |
| Reduce + WHT (40 heads) | ~162 | -- | 1% |

**Key findings:**
- **1.9x steady-state speedup** from rotated-space attention (eliminates WHT from inner loop)
- Path B (K-scores with tree reduction) dominates at 84% — the dot-product reduction (7 barrier stages) costs almost as much as the eliminated WHT butterfly
- Path C (V-accumulate) is very efficient at 0.42 ns/vec — sequential accumulation per thread with no barriers
- Estimated ms/token at 14K: **1.6 ms** (down from 3.0 ms baseline) — still above 1.0 ms target
- Phase 0.5 decision: **1.9x is significant but not sufficient alone. Proceed to Phase 2 (fused kernel) where rotated-space + fused attention will combine both wins.**

**Results:** `~/turboquant-sycl/phase05_bench_results.txt`

### V100 Comparison (Definition of Success)

| Metric | B70 | V100 | Delta |
|--------|-----|------|-------|
| Quality | 20/22 (90.9%) | 22/21 (105%) | Different rubric versions (22-check vs 21-check) |
| Decode c=1 @ 14K | **68.6 t/s** | 49.4 t/s | **+39%** |
| KV cache | 388,747 (FP8) | 155,000 | **+151%** |
| Peak concurrent | 820.3 t/s | — | — |
| BWA-MEM2 | Pending scoring | 18/30 (60%) | — |
| Power | ~80W | 250-300W | **~3.5x more efficient** |

### Earlier Results (pre-standardized)

#### vLLM XPU Graph FULL (4K context, 0.85 gpu-mem-util)

| c | Aggregate tok/s | Per-req tok/s | TTFT | TPOT | Wall time |
|---|----------------|--------------|------|------|-----------|
| 1 | 72.91 | 72.91 | 121ms | 13.71ms | 7.13s |
| 2 | 96.72 | ~49.3 | 218ms | ~20ms | 10.59s |
| 4 | 173.26 | ~44.1 | 228ms | ~23ms | 11.82s |
| 8 | 300.16 | ~38.3 | 291ms | ~26ms | 13.65s |

#### Patched vLLM Eager Mode Detail

| Context | c=1 tok/s | c=2 tok/s | c=4 tok/s | c=8 tok/s | TTFT | Concurrency |
|---------|----------|----------|----------|----------|------|-------------|
| 4,096 | 11.18 | 21.65 | 43.32 | 86.36 | 205ms | 29.40x |
| 16,384 | 11.06 | 21.58 | 43.10 | 85.98 | 331ms | 13.36x |
| 32,768 | ~11 (est) | ~22 (est) | ~43 (est) | ~86 (est) | -- | 6.68x |

#### llama.cpp SYCL Prefill

| Test | Speed (tok/s) | Notes |
|------|--------------|-------|
| pp512 | 992.74 ± 5.10 | Prompt processing |
| pp4096 | 933.45 ± 8.42 | |
| pp8192 | 894.96 ± 2.66 | |
| pp14336 | 852.63 ± 1.38 | V100-comparable context |
| tg128 | 57.16 ± 0.15 | +16% faster than V100 target (49.4) |
| tg128 (14K ctx) | 57.98 ± 0.10 | Speed constant across context lengths |

### Qwen3.6-27B AutoRound INT4 (dense)

| Backend | c=1 tok/s | c=4 tok/s | c=8 tok/s | TPOT c=1 | KV Cache | Concurrency @32K |
|---------|----------|----------|----------|----------|----------|-----------------|
| **Native vLLM + XPU Graph** | **27.40** | 28.71 | — | **36.50 ms** | 83,285 | 20.33x |
| LLM-Scaler (FP16 KV) | 16.36 | **57.35** | — | 58.86 ms | 34,304 | 4.01x |
| Native vLLM 2026 eager (FP8 KV) | 9.50 | 28.45 | 55.14 | 105.23 ms | **214,357** | **6.54x** |
| Docker vLLM 2025.3 (FP8 KV) | 9.28 | 17.83 | — | 106.42 ms | 212,992 | 6.50x |

Note: Dense 27B XPU Graph c=4 is compute-bound (all 27B params active per token). MoE benefits much more from XPU Graph because only ~3B params are active.

### Qwen3-14B sym_int4 (LLM-Scaler only)

| c=1 tok/s | c=4 tok/s | c=8 tok/s | c=14 tok/s | TPOT c=1 | KV Cache |
|----------|----------|----------|-----------|----------|----------|
| 30.31 | 115.03 | 190.49 | 284.47 | 32.46 ms | 118,144 |

## MoE Investigation Results

### vLLM MoE on XPU — RESOLVED

**The Python guard patch worked.** Removing `is_xpu()` from `inc_wna16_scheme.py:60-62` enabled
quantized MoE on XPU. The model loads, serves, and produces coherent output.

**Patch:** Single line change — `if current_platform.is_xpu() or current_platform.is_cpu():` changed to `if current_platform.is_cpu():`

**Method chain:** AutoGPTQMoEMethod -> WNA16 XPU backend -> CUTLASS W4A16 grouped GEMM

**Full results:** `~/b70-vllm/results/moe_patched_full_results.md`
**Track 2 diagnosis:** `~/b70-vllm/results/track2_moe_kernel_diagnosis.md`

### vLLM MoE Attempt History

1. Native vLLM upstream (d272418): `KeyError: w2_qweight` — Python guard blocks quantized MoE
2. LLM-Scaler Docker (0.14.0-b8.3.1): `sycl_arch not recognized: 21483225088` — BMG not supported
3. **Patched native vLLM: SUCCESS** — guard removed, model loads and serves at 11 tok/s c=1, 86 tok/s c=8

## Key Findings

1. **XPU Graph FULL mode is a game-changer** — 72.9 tok/s c=1 (6.5x over eager mode). Eliminates Python/framework overhead by recording and replaying GPU operations. Exceeds llama.cpp hardware ceiling (57 tok/s) because quantized compute kernels are faster than GGML Q4_K_M.

2. **820 tok/s aggregate at c=64** — throughput scales 11.5x from c=1 to c=64 and is still climbing. Sweet spot is c=8–16 for interactive (167–297 t/s), c=48–64 for batch throughput (748–820 t/s). Per-request decode degrades sub-linearly as GPU compute saturates.

3. **XPU Graph requires specific flags** — must use `VLLM_XPU_ENABLE_XPU_GRAPH=1`, `-cc.mode=0` (disable torch.compile), `-cc.cudagraph_mode=full`, `--max-num-seqs 128` (Mamba cache constraint), and must NOT use `--enforce-eager`.

4. **Piecewise mode fails** — AutoRound's C++ `woqgemm` extension is not traceable by torch.dynamo. Only FULL graph capture mode works (records at GPU level, bypasses Python tracing).

5. **llama.cpp SYCL is no longer the performance ceiling** — vLLM + XPU Graph at 73 tok/s exceeds llama.cpp at 57 tok/s, with full serving capability (concurrent requests, streaming, KV cache).

6. **Eager mode bottleneck was 100% framework overhead** — per-token timing showed uniform 89.9ms with 5.6ms stdev. XPU Graph reduced this to 13.7ms, proving the GPU kernels were always fast.

7. **torch.compile doesn't work on XPU** — inductor has no XPU support. But XPU Graph FULL mode works independently.

8. **sym_int4 is broken on 27B models** (race condition in `resadd_norm_gemv_int4` when N > 512).

9. **Rock-solid stability** — 30-minute sustained load at c=4 completed 1,224 requests with 0 errors. Throughput drift was +0.5% (warmup only), confirming no thermal throttling or memory leaks.

10. **B70 exceeds V100 on decode by 39%** — 68.6 t/s at 14K context vs V100's 49.4 t/s. With 2.5x the KV cache (389K vs 155K tokens) and ~3.5x better power efficiency (~80W vs 250–300W).

11. **TurboQuant dequant is ALU-bound, not bandwidth-bound** — Phase 0 microbenchmark shows global FP32 write is free (Mode 4 faster than Mode 5 by 26%). The bottleneck is the 7-stage WHT butterfly with barriers (2.79 ns/vector). Split dequant adds 3.1 ms/token at 14K context (>1.0 ms threshold), so fused kernel (Phase 2) is the priority path. SYCL GPU produces bit-identical results to CPU reference.

12. **Concurrency × context interaction is multiplicative** — Per-request decode degrades ~2x from c=1→c=8 and ~1.4x from 4K→128K context. The practical multi-user sweet spot is c=4@16K (32.8 tok/s, 6s TTFT). Long context (64K+) is only viable at c=1–2. The FP8 KV budget (388K tokens) is the hard ceiling — TurboQuant's 3.8x compression would shift the entire matrix right by one column (e.g., c=4@64K becomes possible).

13. **Rotated-space attention gives 1.9x speedup over split dequant** — Phase 0.5 shows computing attention in Hadamard-rotated space eliminates WHT from the per-token inner loop, reducing dequant+attention cost from 2.71 ns/vec to 1.43 ns/vec (1.93x at steady state). K-score dot-product reduction dominates (84% of Path D cost at 2.35 ns/vec). V-accumulate is extremely cheap (0.42 ns/vec, no barriers). Estimated ms/token at 14K drops from 3.0 to 1.6 ms — significant but still above 1.0 ms target, confirming fused kernel (Phase 2) is needed.

## Disk Usage

| Path | Size | What |
|------|------|------|
| ~/pytorch-src | 15 GB | PyTorch v2.12.0 source + build artifacts |
| ~/vllm-src | 4.5 GB | vLLM source (editable install) |
| ~/vllm-xpu-kernels-src | 4.3 GB | vllm-xpu-kernels source + build |
| ~/auto-round-src | 1.6 GB | auto-round-lib source |
| ~/torchvision-src | 39 MB | torchvision source |
| ~/vllm-native | ~5 GB | Python venv |
| ~/llm-models (NVMe) | ~42 GB | MoE AutoRound (21 GB) + GGUF (21 GB) |
| ~/llama.cpp | ~2 GB | llama.cpp source + build (SYCL) |
| ~/llama-cpp-turboquant | ~1.1 GB | TurboQuant fork + Vulkan build |
| ~/turboquant-sycl | ~100 MB | TurboQuant SYCL Phase 0 microbenchmark |
| /mnt/flash/llm-models | ~46 GB | Dense 27B (18 GB) + 14B (28 GB) |
| /opt/intel/oneapi | ~30 GB | oneAPI 2025.3 + 2026.0 |
| **NVMe used** | **~146 GB / 233 GB** | **~87 GB free** |
| **Flash used** | **~46 GB / 256 GB** | **~188 GB free** |

## Log Files

### Standardized Benchmarks (llm-bench v3.0)
- `~/llm-bench/results/b70-moe-xpugraph-v3.0-20260621-235506/` — full 4-leg benchmark (quality, decode, parallel, context)
- `~/llm-bench/results/b70-moe-parallel-sweep-v3.0-20260622-005131/` — extended parallel sweep (c=1–64)
- `~/llm-bench/results/b70/bwamem2_response.txt` — BWA-MEM2 domain test raw response
- `~/b70-vllm/results/stability/c4_30min_stability.json` — 30-minute stability test summary

### Phase 4 Results
- `~/llm-bench/results/b70/moe_xpugraph_default_kv.txt` — FP8 vs default KV comparison
- `~/llm-bench/results/b70/decode_32k_context.txt` — 32K context decode bench
- `~/llm-bench/results/b70/decode_64k_context.txt` — 64K context decode bench
- `~/llm-bench/results/b70/turboquant_vulkan_baseline.txt` — TurboQuant Vulkan baseline
- `~/llm-bench/results/b70/turboquant_vulkan_turbo4.txt` — TurboQuant turbo4 results
- `~/llm-bench/results/b70/turboquant_vulkan_turbo3.txt` — TurboQuant turbo3 results
- `~/b70-vllm/results/moe_xpugraph_32k.log` — 32K context server log
- `~/b70-vllm/results/moe_xpugraph_64k.log` — 64K context server log

### Phase 0 Results (TurboQuant SYCL)
- `~/turboquant-sycl/bench_results_full.txt` — full benchmark output (modes 1-5, all scales)
- `~/turboquant-sycl/src/` — SYCL kernels and correctness tests
- `~/turboquant-sycl/reference/` — CPU reference implementation

### Phase 0.5 Results (Rotated-Space Attention)
- `~/turboquant-sycl/phase05_bench_results.txt` — full Phase 0.5 benchmark (Paths A-D, all scales)
- `~/turboquant-sycl/src/tq_rotated_bench.cpp` — SYCL rotated-space attention kernels + benchmark
- `~/turboquant-sycl/reference/tq_cpu_attention.h` — CPU attention reference (standard + rotated)

### Concurrency × Context Matrix
- `~/b70-vllm/results/matrix/matrix_results.txt` — raw matrix benchmark results
- `~/matrix_bench.py` — benchmark script

### Earlier Results
All saved to `~/b70-vllm/results/` including:
- `llamacpp_sycl_q4km_bench.log` — initial llama.cpp benchmark
- `llamacpp_sycl_q4km_full_bench.md` — full benchmark report
- `track2_moe_kernel_diagnosis.md` — vLLM MoE kernel investigation
- `moe_patched_full_results.md` — MoE guard patch validation
- 53+ earlier log files covering all phases of testing

## What's Not Working

| Issue | Root Cause | Workaround |
|-------|-----------|------------|
| vLLM quantized MoE on XPU | Python guard in inc_wna16_scheme.py | **FIXED** — guard patched, MoE serving works |
| LLM-Scaler MoE on BMG | sycl_arch not recognized | None (Docker too old) |
| vLLM MoE c=1 speed (11 tok/s) in eager | Python/framework overhead (89.9ms/tok) | **FIXED** — XPU Graph FULL mode (13.7ms/tok, 73 tok/s) |
| sym_int4 garbage on 27B models | Race condition in GEMV kernel (N > 512) | Use AutoRound INT4 pre-quant |
| torch.compile on XPU | Inductor has no XPU backend | `-cc.mode=0` (disable compile, keep XPU Graph) |
| XPU Graph PIECEWISE | AutoRound woqgemm not dynamo-traceable | Use FULL mode (`-cc.cudagraph_mode=full`) |
| Triton attention backend | KV cache page size incompatibility with GDN layers | Use Flash Attention v2 (default) |
| XCCL distributed | oneCCL Level Zero init fails natively | Falls back to gloo (OK for 1 GPU) |
| Thinking mode on long prompts | Model spends entire token budget in `<think>`, null content | Disable thinking (`enable_thinking=false`) for comprehension tests |
| BWA-MEM2 repetition loop | INT4 quantization degrades complex domain reasoning | Score manually; may improve with higher quant |
| BF16 serving | XPU backend only supports FP16 | Cast to FP16 (automatic) |
| FP8 online quantization | Loads FP16 weights first (OOM on 27B) | Use pre-quantized models |

## Quick Start

```bash
# SSH to rig
ssh user@192.168.1.57  # password: 1

# llama.cpp benchmark (fastest)
source /opt/intel/oneapi/setvars.sh --force
~/llama.cpp/build/bin/llama-bench -m ~/llm-models/gguf/Qwen_Qwen3.6-35B-A3B-Q4_K_M.gguf -ngl 999

# llama.cpp server
~/llama.cpp/build/bin/llama-server -m ~/llm-models/gguf/Qwen_Qwen3.6-35B-A3B-Q4_K_M.gguf -ngl 999 --port 8080

# Start vLLM server — MoE with XPU Graph (fastest, 73 tok/s c=1)
nohup bash ~/serve-moe-xpugraph-full.sh > ~/b70-vllm/results/serve.log 2>&1 &

# Start vLLM server — MoE eager mode (slower but more context)
nohup bash ~/serve-moe.sh > ~/b70-vllm/results/serve.log 2>&1 &

# Start vLLM server — Dense 27B
nohup bash ~/serve-native.sh > ~/b70-vllm/results/serve.log 2>&1 &

# Test vLLM
curl http://localhost:8000/v1/chat/completions \
  -H "Content-Type: application/json" \
  -d '{"model":"Qwen3.6-35B-A3B","messages":[{"role":"user","content":"Hello"}],"max_tokens":100}'

# Stop vLLM
pkill -f 'vllm serve'

# Check GPU
xpu-smi discovery
```
