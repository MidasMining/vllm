# TurboQuant SYCL — Architectural Decision Document (v2)

*Date: 2026-06-22*
*Revised after Claude ↔ GPT architectural review*
*Context: Deciding the kernel approach for TurboQuant SYCL on Intel Arc Pro B70*

---

## 1. What We're Building

A SYCL implementation of TurboQuant KV cache compression for Intel Arc GPUs, targeting vLLM (vllm-xpu-kernels) as the Pearl-relevant path. TurboQuant compresses KV cache vectors from FP16 to 3-4 bits using:

1. **Walsh-Hadamard Transform (WHT)** — rotate each KV vector by a fixed random orthogonal matrix (128-element butterfly, 7 stages)
2. **Lloyd-Max codebook quantization** — quantize each rotated coordinate to nearest centroid (pre-computed, compile-time constants)
3. **Index packing** — pack centroid indices into compact storage (3 or 4 bits per coordinate)
4. **Dequantization** — unpack indices, look up centroids, apply inverse rotation (during attention)

The critical path is step 4 (dequant during attention decode). Steps 1-3 happen once when a KV vector is written to cache. Step 4 happens on every decode step for every cached token — it's in the hot loop.

---

## 2. What We Know About B70 Kernel Performance

`MidasMining/alphamine-intel` contains two separate investigation tracks. These must not be conflated.

### MKL investigation (hardware ceiling reference)

The oneMKL GEMM benchmarks (`mkl_gemm_bench.cpp`, `mkl_sustained_bench.cpp`) demonstrate what the B70 hardware can achieve when the workload maps cleanly to an optimized library GEMM shape. This proves headroom — the card is not inherently the bottleneck — but does not inform custom kernel architecture because TurboQuant is not a clean GEMM problem.

**MKL tells us what the B70 can theoretically do. It does not tell us how to build TurboQuant.**

### ESIMD miner investigation (architectural precedent)

The live production miner (`b70_miner.cpp`, `xe_esimd_gemm*.cpp`) uses custom SYCL ESIMD kernels. 50+ variants were tested. The best live performance continues to come from ESIMD, not MKL. This is the relevant analogy for TurboQuant because TurboQuant integration is a custom fused memory/ALU/attention problem, not a standard GEMM call.

**ESIMD tells us what happens when we own the fused/custom kernel and have to fight IGC, register pressure, occupancy, memory layout, and pipeline overlap.**

### Hardware constraints (verified on B70 via ESIMD investigation)

| Constraint | Value | Impact |
|-----------|-------|--------|
| GRF file | 128 registers (small mode) optimal | 256-GRF halves threads/EU, kills occupancy |
| Optimal accumulators | 8 (fills 128-GRF exactly) | More → spill; fewer → wasted capacity |
| DPAS execution size | SIMD16 mandatory on Xe2 | SIMD8 (A750 patterns) → OOB GRF → hang |
| Threads/EU | 8 (small GRF mode) | Occupancy is the primary performance lever |
| SLM per subslice | 131072 bytes | Adequate for butterfly + codebook |
| Peak INT8 | 183 TMAC/s (367 TOPS) | Production ESIMD miner achieves 67.72 TMAC/s (37%) |

### Key ESIMD miner findings

**Cross-thread ALU/DPAS overlap:** The fused miner kernel (GEMM+fold+BLAKE3) at 67.72 TMAC/s beats the split two-pass kernel at 51.38 TMAC/s. With 8 threads per EU at different pipeline stages, DPAS and vector ALU stay utilized simultaneously when both types of work coexist in a single kernel.

**Register pressure ceiling:** The fused miner's bottleneck is the `trans[16]` transcript array — 16 GRFs that must remain live across the K-loop. Any consumer that needs the individual 16 values prevents IGC from optimizing them away, degrading performance from 174 to 68 TMAC/s (2.6x penalty). Replacing the consumer with a commutative reduction lets IGC collapse the array to a single accumulator, recovering full speed.

**Workgroup geometry matters as much as fusion:** The dominant performance wins came from tile shape (cm64/cn32), local_size tuning (32 best), occupancy management (8 accumulators = 128-GRF sweet spot), and memory-level parallelism — not just the fused/split decision. The fused advantage is real, but it is not the only lever.

**The corrected architectural lesson:** The decisive B70 kernel pattern is not simply "fused beats split." It is: **keep data local, avoid global memory round-trips, preserve 128-GRF occupancy, and avoid long-lived register arrays.** That is the design principle TurboQuant should follow.

### IGC compiler limitations (verified on oneAPI 2026.0)

| Limitation | Impact |
|-----------|--------|
| Noinline ABI broken | Data arrives as zeros across noinline boundary — cannot split hot functions |
| Double-buffer K-loop | ICE (FP exception) during JIT — compiler crash |
| lsc_load_2d prefetch | Not instantiated on Battlemage (DG2/PVC only) |
| 256-GRF mode | Halves threads/EU, occupancy loss > blocking gain (102→38 TMAC/s) |
| Atomic DCE across noinline | Even `atomic_update<xchg>` eliminated across noinline boundary |

---

## 3. The Architectural Options

### Option A: Fused Dequant+Attention (production target)

Integrate TurboQuant dequantization directly into the flash attention SYCL kernel. During each decode step, the kernel loads packed TurboQuant indices, dequants, and feeds reconstructed values into attention computation within the same kernel and register file.

**Critical design rule: No long-lived reconstructed vector arrays across the attention inner loop.**

The fused kernel must NOT do this:

```
unpack full 128-dim vector → dequant full vector → inverse WHT full vector → keep reconstructed FP16 vector live → feed attention
```

That risks reproducing the miner's `trans[16]` register pressure problem. Instead:

```
packed indices → small chunk decode → local WHT butterfly fragment → immediately consume into QK or V accumulation
```

Dequant must be **producer-consumer local**. Do not create a long-lived `k_vec[128]` or `v_vec[128]` unless the compiler proves it scalarizes well. On B70/IGC, assume it will not.

**Pros:**
- Cross-thread ALU/DPAS overlap potential
- No intermediate FP16 buffer (saves bandwidth — critical on a 608 GB/s card)
- No extra kernel launch overhead
- Matches the llama.cpp TurboQuant Vulkan approach (fused via GGML_OP_TURBO_WHT)
- Avoids the global memory expansion that TurboQuant is specifically designed to prevent

**Cons:**
- Flash attention integration is the hard part — vllm-xpu-kernels uses CUTLASS XE2 templates, different from Vulkan spec constants
- Register pressure risk if chunk-local consumption is not achieved
- Longer development cycle
- IGC compiler limitations reduce optimization options

**Register pressure estimate:**
- TurboQuant turbo4 codebook: 2^4 = 16 centroids × FP16 = 32 bytes (2 GRFs, trivial)
- WHT rotation: procedural (±1 signs, no storage — butterfly stages use sign-multiply)
- Per-chunk reconstructed values: depends on chunk size (target: small enough to consume immediately)
- QK^T partial products: depends on tile size
- Working registers for butterfly stages: ~8-16 GRFs

Fits in 128-GRF if tile size is conservative and chunk-local consumption is maintained.

### Option B: Split Dequant-Then-Attend (correctness oracle)

Standalone TurboQuant dequant kernel that reads packed KV cache, reconstructs full FP16 vectors, and writes them to a temporary global memory buffer. Existing flash attention reads FP16 from the buffer.

**Role: correctness oracle and overhead measurement baseline. Not the expected final production design.**

Split dequant materializes expanded FP16 KV to global memory — exactly the thing TurboQuant is designed to avoid. At short contexts (14K) the ~6.5% bandwidth overhead may be tolerable. At long contexts (60K, 128K) or high concurrency, the extra global round-trip erases much of TurboQuant's practical advantage. Since the B70's value proposition is large KV capacity and long-context serving, a split implementation would undermine the core use case.

**Bandwidth estimate (split):**
At 14K context, c=1 decode, d_head=128, n_heads=40 (Qwen3.6):
- KV read (packed turbo4): ~70 MB per decode step
- KV dequant write (FP16): ~280 MB
- KV attention read (FP16): ~280 MB
- Total extra from split: ~280 MB write+read = ~0.9 ms at 608 GB/s
- At 14K: ~6.5% overhead on current 13.7 ms/token
- At 60K: ~4× more KV, proportionally worse
- At 128K: overhead becomes a significant fraction of token time

XPU Graph FULL reduces launch overhead but does **not** eliminate the global FP16 materialization cost.

**Pros:**
- Simple to build (~1 day)
- Independently testable, easy to debug
- Existing flash attention works unmodified
- No register pressure interaction
- Provides correctness reference for validating fused implementation

**Cons:**
- Extra global memory bandwidth (the dominant concern for long-context)
- No cross-thread ALU/DPAS overlap
- Undermines TurboQuant's core purpose at long contexts
- Not suitable as a final production design for B70 long-context serving

### Option C: SLM Hybrid (fallback only)

Dequant into SLM, attention reads from SLM. Demoted to fallback status.

SLM capacity (128 KB) with K+V and alignment/padding holds ~256 token positions per tile, requiring multiple tiled passes for long context plus barriers between dequant and attention phases. This is effectively a custom tiled attention rewrite rather than a simple compromise between split and fused.

**Use only if:** fused direct consumption has bad register pressure AND split global-memory overhead is too high.

---

## 4. What The Miner Experience Tells Us (Corrected)

The alphamine ESIMD miner is the relevant precedent for TurboQuant because both involve custom fused kernel work on B70 — not library GEMM calls.

### What transfers

- **Fusion can overlap ALU with DPAS**, but the magnitude depends on the ALU/DPAS ratio and tile geometry, not just the decision to fuse.
- **Global memory round-trips hurt.** The miner's two-pass split lost ~32% partly because of transcript store/load traffic. TurboQuant's split path has the same structural problem (expanded FP16 KV write+read).
- **128-GRF occupancy is the primary performance lever.** 256-GRF mode hurts. 8 accumulators is the sweet spot.
- **Register live ranges must be kept short.** The `trans[16]` problem (16 GRFs live across the K-loop) is the miner's fundamental bottleneck. TurboQuant must not create an equivalent with full reconstructed K/V vectors.
- **IGC has sharp edges.** Noinline ABI broken, double-buffer ICE, prefetch missing. Plan for compiler constraints upfront.
- **Tile shape and load patterns matter.** The miner's performance progression (28→54→76→91→103→108 TMAC/s) came mostly from geometry and memory tuning, not from the fused/split decision alone.

### What is different for TurboQuant

- The miner's hot path is a GEMM K-loop. TurboQuant's hot path is an attention loop over cached tokens. Different iteration structure.
- The miner's register pressure comes from a 16-GRF transcript array. TurboQuant's codebook is 2 GRFs — much smaller, if chunk-local consumption avoids creating equivalent long-lived arrays.
- The miner's BLAKE3 is 700 ALU instructions (54% of kernel). TurboQuant's WHT dequant is lighter per vector. The ALU/DPAS overlap benefit may be proportionally smaller.

### The clean architectural principle

For B70 TurboQuant:

- **MKL proves headroom; ESIMD proves the route.**
- Custom SYCL/ESIMD kernels win on B70 when they preserve locality, avoid unnecessary global round-trips, maintain 128-GRF occupancy, and keep register live ranges short.
- TurboQuant should not be expressed through MKL. The production path is custom SYCL integration where packed KV is dequantized close to where attention consumes it.

---

## 5. Implementation Plan

### Recommended phased approach

```
Phase 0: Standalone TurboQuant SYCL correctness + microbenchmark harness
Phase 1: Split dequant-then-attend baseline (correctness oracle)
Phase 2: Fused dequant-inside-attention (production target)
Phase 3: SLM/tiled hybrid only if fused hits register/IGC cliffs
```

**The split path (Phase 1) is a correctness oracle and overhead measurement tool, not the expected final B70 performance path.** Final production target is fused dequant inside attention (Phase 2).

### Phase 0 — Standalone microbenchmark

Build a standalone SYCL TurboQuant correctness and performance harness, independent of vLLM or llama.cpp:

**Correctness tests:**
- Pack FP16 → turbo4 → unpack → verify round-trip error matches paper MSE
- Pack FP16 → turbo3 → unpack → verify
- Codebook lookup correctness against reference implementation
- WHT butterfly correctness (forward + inverse should recover original within quantization error)

**Microbenchmark modes:**
- turbo4 unpack only
- turbo4 unpack + codebook lookup
- turbo4 unpack + codebook + inverse WHT
- turbo3 equivalents
- With FP16 global memory write (measures bandwidth cost)
- Without FP16 write / checksum-only mode (measures pure ALU/WHT cost)

**Key measurement:** Compare dequant-to-global-write vs dequant-checksum-only. The difference is the global write bandwidth cost — this directly determines whether split (Option B) is tolerable or whether fused is mandatory.

**Threshold:** If standalone split dequant costs < 0.5–1.0 ms/token at 14K, Option B is acceptable as a baseline. If it costs several ms/token, skip directly to fused.

**Phase 0 acceptance criteria:**

Phase 0 is complete when:
1. turbo3/turbo4 pack/unpack/codebook/WHT match CPU reference within expected quantization error.
2. Checksum-only dequant and dequant-to-FP16-global-write modes both run.
3. Report includes GB/s effective packed read, GB/s FP16 write, vectors/s, ns/vector, and estimated ms/token at 14K, 60K, and 128K.
4. Results are captured for at least d_head=128 and Qwen3.6-style head counts (40 heads).
5. Kernel compiles in 128-GRF mode without spills or forced 256-GRF mode.

**Overhead threshold:** If split dequant is under ~0.5–1.0 ms/token at 14K and scales acceptably at 60K+, it is acceptable as a temporary baseline. If it is several ms/token or scales linearly into a large fraction of decode time, skip further split work and prioritize fused.

### Phase 1 — Split baseline

Build split dequant as a standalone kernel + existing flash attention. This provides:
- Correctness oracle for validating Phase 2
- End-to-end overhead measurement under real serving conditions
- Quick integration path if overhead is surprisingly low

### Phase 2 — Fused production kernel

Build fused dequant-inside-attention following the chunk-local consumption pattern:

```
packed indices → small chunk decode → local WHT butterfly fragment → immediately consume into QK or V accumulation
```

Use Phase 1 as the correctness reference — fused output should match split output within floating-point tolerance.

### Phase 3 — SLM fallback (only if needed)

Only if Phase 2 hits register pressure or IGC problems that prevent chunk-local consumption.

### Integration target order

```
1. Standalone SYCL pack/dequant/WHT correctness test (no framework dependency)
2. Standalone split dequant benchmark: packed KV → FP16 temp
3. vLLM-side split integration if the hook is straightforward
4. Fused vLLM/XPU attention prototype
5. llama.cpp SYCL only if it materially speeds iteration or gives easier validation
```

Start with standalone SYCL, not inside any framework. The core kernel logic (WHT butterfly, codebook lookup, index packing/unpacking) is framework-independent. Integration wrappers come after correctness is proven.

Do not start with llama.cpp just because it is easier unless most kernel code would be reused unchanged. The integration surface differs enough between llama.cpp and vLLM that it may become a detour from the Pearl-relevant target.

---

## 6. Decision Summary

| Factor | Fused (A) | Split (B) | Hybrid SLM (C) |
|--------|-----------|-----------|-----------------|
| Role | Production target | Correctness oracle | Fallback |
| Expected speed | Best (locality + overlap) | Acceptable short-ctx, poor long-ctx | Middle |
| Register pressure risk | Medium (chunk-local mitigates) | None | Low |
| Development time | 2-3 days | 1 day | 1.5 days |
| Flash attention integration | Required (hard) | Not needed (easy) | Partial rewrite |
| Bandwidth overhead | None | ~6.5% at 14K, much worse at 60K+ | Minimal (SLM) |
| IGC risk | Higher (complex kernel) | Lower | Medium |
| Long-context viability | Yes | Poor (expands KV to FP16 globally) | Moderate |
| Build order | Phase 2 | Phase 1 | Phase 3 (if needed) |

---

## 7. Resolved Questions

**1. Is the fused advantage transferable from the miner?**
Partially. The miner proves fusion and occupancy can hide mixed ALU/DPAS work, but the dominant wins came from geometry, memory-level parallelism, and load shape — not just ALU/DPAS overlap. For TurboQuant, the first design goal is not merely "fuse because fusion won." It is: avoid materializing full FP16 KV to global memory while keeping register lifetime short and tile geometry friendly to Xe2 occupancy.

**2. Does XPU Graph FULL change the split calculus?**
It reduces launch overhead but does NOT eliminate the global FP16 materialization cost. For split TurboQuant, bandwidth is the bigger long-context concern than launch overhead.

**3. Which target first?**
Standalone SYCL core first, then vLLM split/fused. llama.cpp only if it materially speeds iteration.

**4. Does fused fit in 128 GRF?**
Likely yes, but only if the implementation avoids full reconstructed K[128]/V[128] live arrays. Chunk-local consumption is the design requirement. If it materializes whole vectors, expect IGC register pressure problems.

**5. Should dequant double as Pearl mining?**
Keep out of the first integration. WHT is structured ±1 work, not dense DPAS-friendly GEMM. Treat inference-mining overlap as a separate experiment after TurboQuant performance is proven.

---

## 8. Source References

| Source | What it provides |
|--------|-----------------|
| `MidasMining/alphamine-intel` (ESIMD track) | B70 kernel architecture, register/occupancy constraints, IGC limitations, cross-thread overlap evidence |
| `MidasMining/alphamine-intel` (MKL track) | Hardware ceiling reference only — not the implementation path |
| `TheTom/llama-cpp-turboquant` (Vulkan) | Working TurboQuant implementation on B70 Vulkan; code review for algorithm reference |
| `vllm-project/vllm-xpu-kernels` | Flash Attention v2 SYCL integration target; CUTLASS XE2 kernel infrastructure |
| `vllm-project/vllm` PR #38479 | Upstream TurboQuant Python config, presets, KV cache management |
| Google TurboQuant paper (ICLR 2026) | Algorithm specification, WHT butterfly, Lloyd-Max codebooks, distortion bounds |

---

*This document reflects the agreed architectural approach after Claude ↔ GPT review. Phase 0 (standalone microbench) should begin first. No flash attention integration until correctness and overhead measurements are complete.*
