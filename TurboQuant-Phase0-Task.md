# CC Task: TurboQuant SYCL Phase 0 — Standalone Correctness & Microbenchmark

*Date: 2026-06-22*
*Architecture basis: TurboQuant-SYCL-Architecture.md v2 (approved)*
*Context: Phase 0 of a 4-phase TurboQuant SYCL implementation. This phase validates correctness and measures cost independent of any serving framework.*

---

## Objective

Build a standalone SYCL TurboQuant correctness and microbenchmark harness that validates packing, codebook lookup, inverse WHT, and FP16 reconstruction on the B70. This harness is framework-independent — it does not touch vLLM, llama.cpp, or any serving stack.

The critical measurement is: **how much of the dequant cost is memory materialization versus ALU/WHT math?** This determines whether the split approach (Phase 1) is tolerable or whether we need to jump directly to fused (Phase 2).

---

## Architecture Reference

The architectural decisions are documented in `TurboQuant-SYCL-Architecture.md v2`. Key points for this phase:

- **MKL is NOT the implementation strategy.** MKL is the hardware ceiling reference. The relevant kernel development precedent is the ESIMD miner in `alphamine-intel`.
- **Target 128-GRF mode.** 256-GRF halves threads/EU and kills occupancy (verified: 102→38 TMAC/s in the miner).
- **DPAS execution size must be SIMD16 on Xe2.** SIMD8 causes OOB GRF → hang.
- Use `ONEAPI_DEVICE_SELECTOR=level_zero:gpu` (the rig also has an AMD iGPU).
- Source `/opt/intel/oneapi/setvars.sh` before building.

---

## Algorithm Reference

TurboQuant per-vector operations (for d_head=128):

### Quantization (KV write path — less critical, happens once per token)

```
input: FP16 vector v[128]
1. compute per-vector norm: s = ||v||
2. normalize: v_hat = v / s
3. apply Walsh-Hadamard Transform: v_rot = WHT(v_hat)
   - 7 butterfly stages (log2(128) = 7)
   - each stage: for i in range(128): v_rot[i] = ±v_rot[j] ± v_rot[k] (structured signs)
4. quantize each coordinate: idx[i] = nearest_centroid(v_rot[i], codebook)
   - codebook: 16 centroids for turbo4 (4 bits), 8 for turbo3 (3 bits)
   - Lloyd-Max optimal centroids, pre-computed, compile-time constants
5. pack: pack idx[128] into compact storage (64 bytes for turbo4, 48 bytes for turbo3)
6. store: packed indices + norm scalar
```

### Dequantization (KV read path — critical hot loop during attention)

```
input: packed indices + norm scalar
1. unpack: extract idx[128] from packed storage
2. codebook lookup: v_rot[i] = codebook[idx[i]] for each coordinate
3. inverse WHT: v_hat = WHT_inverse(v_rot)
   - same 7 butterfly stages, inverse signs
4. denormalize: v = v_hat * s
output: reconstructed FP16 vector v[128]
```

### Lloyd-Max codebook (compile-time constants)

For turbo4 (4-bit, 16 centroids), the codebook values are pre-computed from the Beta distribution for d=128 after Hadamard rotation. These are fixed constants — same for every model, every layer, every head. The paper provides the exact values; the llama.cpp TurboQuant fork has them implemented.

**Source reference for codebook values and WHT implementation:** `~/llama-cpp-turboquant/ggml/src/` — search for `turbo`, `hadamard`, `codebook`, `lloyd_max`.

---

## What To Build

### Project structure

```
~/turboquant-sycl/
  CMakeLists.txt
  src/
    tq_types.h          # turbo3/turbo4 packed type definitions
    tq_codebook.h       # Lloyd-Max centroids (compile-time constants)
    tq_wht.h            # Walsh-Hadamard Transform (forward + inverse)
    tq_pack.h           # pack FP16 → turbo indices
    tq_unpack.h         # unpack turbo indices → codebook values
    tq_dequant.h        # full dequant pipeline (unpack + codebook + WHT + denorm)
    tq_bench.cpp        # microbenchmark harness
    tq_correctness.cpp  # correctness tests against CPU reference
  reference/
    tq_cpu_reference.h  # pure C++ CPU reference implementation (no SYCL)
```

### Step 1: CPU reference implementation

Write a pure C++ (no SYCL) reference implementation of:
- WHT forward and inverse (butterfly, exact signs from the Hadamard matrix)
- Lloyd-Max codebook for turbo4 (16 centroids) and turbo3 (8 centroids)
- Pack and unpack
- Full quantize and dequantize pipeline

Use this as the correctness oracle. Extract codebook values and WHT sign patterns from the llama.cpp TurboQuant fork (`~/llama-cpp-turboquant/`).

### Step 2: SYCL dequant kernel

Write a SYCL kernel that performs dequantization on the GPU. Start with standard SYCL (not ESIMD) for correctness, then optimize with ESIMD if needed.

**Six benchmark modes:**

| Mode | Operations | What it measures |
|------|-----------|-----------------|
| 1. Unpack only | Extract indices from packed storage | Bit manipulation overhead |
| 2. Unpack + codebook | Indices → centroid lookup | Memory/table lookup cost |
| 3. Unpack + codebook + inverse WHT | Full dequant pipeline, no output write | Pure ALU/WHT cost |
| 4. Full dequant + FP16 global write | Dequant → write reconstructed FP16 to global memory | **Split approach cost** |
| 5. Full dequant + checksum only | Dequant → reduce to scalar checksum (no global write) | ALU-only cost, no bandwidth |
| 6. Pack (quantize) path | FP16 → WHT → codebook → pack | Write-path cost (less critical) |

**The critical comparison is Mode 4 vs Mode 5.** The difference is the global FP16 write bandwidth cost — this directly determines whether split dequant (Phase 1) is tolerable.

### Step 3: Correctness validation

For each mode, compare SYCL output against CPU reference:
- Generate random FP16 vectors (d=128)
- Pack on CPU, dequant on GPU, compare against CPU dequant
- Report max absolute error and MSE
- Verify MSE matches paper values (turbo4: MSE ≈ 0.009, turbo3: MSE ≈ 0.034)

### Step 4: Microbenchmark

Run all six modes at varying scales:

| Parameter | Values |
|-----------|--------|
| d_head | 128 (Qwen3.6) |
| n_heads | 40 (Qwen3.6) |
| n_tokens | 1000, 4000, 14000, 60000, 128000 |
| Quant type | turbo4, turbo3 |
| Iterations | 100 (warmup 10) |

**Report for each configuration:**

```text
mode:
quant_type:
n_tokens:
n_heads:
d_head:
total_vectors: (n_tokens × n_heads × 2 for K+V)
packed_bytes:
time_us:
vectors_per_second:
ns_per_vector:
GB_s_packed_read:
GB_s_fp16_write: (mode 4 only)
estimated_ms_per_token_14K:
estimated_ms_per_token_60K:
estimated_ms_per_token_128K:
GRF_mode: (128 or 256)
spill_count: (from compiler output if available)
```

---

## Build Instructions

```bash
mkdir -p ~/turboquant-sycl
cd ~/turboquant-sycl

# Source oneAPI
source /opt/intel/oneapi/setvars.sh
export ONEAPI_DEVICE_SELECTOR=level_zero:gpu

# Build with SYCL
cmake -B build \
  -DCMAKE_C_COMPILER=icx \
  -DCMAKE_CXX_COMPILER=icpx \
  -DCMAKE_CXX_FLAGS="-fsycl -fsycl-targets=intel_gpu_bmg_g31" \
  -DCMAKE_BUILD_TYPE=Release

cmake --build build -j $(nproc)
```

If `-fsycl-targets=intel_gpu_bmg_g31` is not recognized, try `-fsycl-targets=spir64_gen -Xs "-device bmg"` or just `-fsycl` for JIT compilation.

Verify 128-GRF mode. If the compiler forces 256-GRF, add `-ftarget-register-alloc-mode=pvc:default` (or the BMG equivalent) to stay in small-GRF mode.

---

## Extracting Reference Data from llama.cpp TurboQuant

The codebook values and WHT implementation exist in the llama.cpp TurboQuant fork already built at `~/llama-cpp-turboquant/`:

```bash
# Find codebook/centroid definitions
grep -r "centroid\|codebook\|lloyd_max\|lloyd" \
  ~/llama-cpp-turboquant/ggml/src/ \
  --include="*.cpp" --include="*.h" --include="*.inc" -l

# Find WHT/Hadamard implementation
grep -r "hadamard\|wht\|butterfly\|turbo.*quant\|turbo.*dequant" \
  ~/llama-cpp-turboquant/ggml/src/ \
  --include="*.cpp" --include="*.h" --include="*.inc" -l

# Find packed type definitions
grep -r "TURBO\|block_tq\|turbo[234]" \
  ~/llama-cpp-turboquant/ggml/include/ \
  ~/llama-cpp-turboquant/ggml/src/ \
  --include="*.h" --include="*.cpp" -l
```

Extract the codebook constants and WHT sign patterns from these files. They are compile-time constants that should be identical in the SYCL implementation.

---

## Disk Space

Estimated: ~100 MB for source + build. Current free space is ~60 GB. No concern.

---

## What NOT To Do

- Do NOT integrate with vLLM or llama.cpp in this phase. Standalone only.
- Do NOT write ESIMD kernels yet. Start with standard SYCL for correctness. ESIMD optimization is Phase 2+ work.
- Do NOT attempt to fuse with flash attention. That is Phase 2.
- Do NOT use oneMKL for the WHT/dequant. This is a custom kernel problem.
- Do NOT use 256-GRF mode. Stay in 128-GRF for occupancy.
- Do NOT benchmark inside vLLM's serving loop. Standalone microbench only.

---

## Acceptance Criteria

Phase 0 is complete when:

1. turbo3/turbo4 pack/unpack/codebook/WHT match CPU reference within expected quantization error (turbo4 MSE ≈ 0.009, turbo3 MSE ≈ 0.034).
2. Checksum-only dequant (Mode 5) and dequant-to-FP16-global-write (Mode 4) both run correctly.
3. Report includes GB/s effective packed read, GB/s FP16 write, vectors/s, ns/vector, and estimated ms/token at 14K, 60K, and 128K.
4. Results are captured for d_head=128 and n_heads=40 (Qwen3.6 configuration).
5. Kernel compiles in 128-GRF mode without spills or forced 256-GRF mode.
6. Mode 4 vs Mode 5 comparison quantifies the global FP16 write bandwidth cost.
7. All source code and results are committed to a new repo or directory.

---

## Expected Deliverables

| Deliverable | Location |
|-------------|----------|
| Standalone TurboQuant SYCL project | `~/turboquant-sycl/` |
| CPU reference implementation | `~/turboquant-sycl/reference/` |
| Correctness test results | In CC's response + saved to project dir |
| Microbenchmark results (all 6 modes) | In CC's response + saved to project dir |
| Mode 4 vs Mode 5 comparison (the key measurement) | In CC's response |
| Estimated ms/token at 14K, 60K, 128K | In CC's response |
| GRF mode and spill count confirmation | In CC's response |

---

## After Phase 0

The results determine the Phase 1/Phase 2 priority:

| Phase 0 Result | Next Step |
|---------------|-----------|
| Mode 4 (split) adds < 0.5 ms/token at 14K, scales acceptably | Proceed to Phase 1 (split baseline as correctness oracle), then Phase 2 (fused) |
| Mode 4 (split) adds > 1.0 ms/token at 14K or scales poorly | Skip Phase 1, proceed directly to Phase 2 (fused is mandatory) |
| Correctness fails (MSE far from paper values) | Debug algorithm — check codebook values, WHT signs, packing format |
| 256-GRF forced or spills detected | Investigate register pressure — may need to restructure kernel before proceeding |

---

*This is Phase 0 only. Do not begin flash attention integration or vLLM integration until Phase 0 is complete and reviewed.*
