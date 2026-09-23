# TQ-for-GLM: integration design (investigation phase 1)

Target: TurboQuant KV compression for GLM-5.3-Flash's 11 sparse-MLA attention
layers on the v0.30 tree (glm53-pp4-v0300). Goal: attention pages 2x (k8v4)
then ~5x (3bit_nc); net pool gain capped by incompressible mamba pages.
Serving today: 634k tokens @ 256k; attention pages are 3x4.72MB of the
14.6MB/block core cost, so k8v4 ≈ -7MB/block → pool ≈ +90%-ish on attention-
bound ranks (exact after page-equalization math).

## Ground truth mapped (2026-09-22)
1. Upstream v0.30 TQ surface: --kv-cache-dtype turboquant_{k8v4,4bit_nc,
   k3v4_nc,3bit_nc} are first-class CacheDTypes (config/cache.py); KVQuantMode
   on specs; TurboQuantAttentionBackend (backends/turboquant_attn.py, 1243
   lines) implements store+decode for STANDARD attention only — no MLA/sparse
   support. GLM sparse layers cannot use it directly.
2. Hybrid hook (_get_full_attention_layer_indices, quantization/turboquant/
   config.py) matches layer_types values ("full_attention","attention") —
   GLM says "deepseek_sparse_attention" -> one-line filter extension needed.
3. GLM sparse read kernel (our port, v1/attention/ops/triton_mla_sparse_
   kernel.py + backends/mla/triton_mla_sparse.py): bf16-only. NO fp8 branch.
   NoPE (qk_rope_head_dim=0 for sparse layers), cache 1152B/token bf16
   (kv_c 512 + 64 = 576 elems; block 4096; mla_page 4.72MB).
4. Donor kernel (CORRECTED after do-all branch audit): the fork's
   mla-sparse-tq-do-all branch has the COMPLETE sparse preset matrix landed
   in triton_mla_sparse_tq_kernel.py — k8v4 AND MSE 4/3/2-bit dequant bodies
   with centroids, plus SM<89 fp8 STORE via fp8e5m2 (commit 88fff22fe).
   Phases 1+2 are therefore a single port of a finished kernel onto GLM's
   slot layout, not kernel authorship. Bonus: feat/path-b-scratch-ampere has
   TQ flash decode ported to Ampere with CMP170HX gating (VLLM_TQ_FLASH_
   DECODE) — the post-TQ decode-perf layer, pre-benched on 8x A4000.
5. Store side: shared mla_attention.py carries fp8_ds_mla store for DSv4;
   GLM writes bf16. Fork has fused TQ store kernels (triton_turboquant_
   store.py, also upstream) + fp8_sm80 software encode helpers (in-tree).
6. Scales: fork k8v4 used checkpoint k_scale (fp8 ckpts). GLM AWQ has no KV
   scales -> need dynamic per-token (or per-block) scales stored alongside,
   fp8_ds_mla-style (u8 payload + fp32/fp16 scale per token).

## Phase 1: k8v4 for GLM sparse layers (the fp8-KV milestone)
A. Spec/plumbing
   - Extend hybrid hook filter (+"deepseek_sparse_attention").
   - MLAAttentionSpec: accept kv_quant_mode k8v4 for glm sparse; page math:
     576 u8 + 2B scale = 578B/token (vs 1152) -> pack/align to 640B?
     Decide alignment with the arena layout (block_stride multiple).
   - supports_kv_cache_dtype("turboquant_k8v4") on TritonMLASparseBackend,
     gated to SM80+GLM path.
B. Store
   - At GLM's cache write: quantize kv_c+extras to e4m3 u8 with per-token
     scale (donor: upstream triton_turboquant_store fp8 path; SM80-safe via
     fp8_sm80 encode pattern — no fp8e4nv stores in Triton on SM80, write u8).
C. Read
   - Port the fork k8v4 dequant tile into our sparse kernel behind a
     TQ constexpr: load u8 -> bitcast fp8e4nv -> f32 * scale -> bf16, per
     576-elem row; keep int64 offsets; split-KV structure unchanged.
   - Scale load per token from the interleaved slot bytes.
D. Validation ladder (one boot each): boots+loads -> pool math check ->
   temp-0 correctness + BWA-MEM2 canary + needle-ish 64k/200k probes ->
   acceptance (drafter unaffected; taps are hidden states not KV) ->
   battery. Quality gate: decode within 2% and 28/30-class domain answer.

## Phase 2: 3bit_nc on sparse (after k8v4 proves the wiring)
   Land the MSE dequant body in the sparse tile from the fork's standard-
   kernel donors (centroids + norm correction); scales/centroid tables via
   the upstream TurboQuantConfig machinery; same validation ladder.

## Open questions before coding
1. Exact GLM slot content beyond kv_c[512]: what are the extra 64 elems on
   NoPE layers (gate? indexer coupling?) — quantization sensitivity unknown;
   may keep those 64 in bf16 (mixed slot: 512 u8 + 64 bf16 + scale = 706B).
2. Per-token vs per-block scale granularity (fork used per-tensor from ckpt;
   fp8_ds_mla uses per-token) — start per-token fp16.
3. Arena/page-equalization interplay: idx/tail pages already 152KB; new mla
   page ~2.5MB keeps divisibility? Verify _glm5_next_tensor_layout math.

## New biggest open question (post-audit)
Centroid provenance: the MSE presets load bf16 centroid tables
(quantization/turboquant/centroids.py). How were the fork's calibrated, and
do they transfer to GLM's kv_c distribution (DSv4-calibrated?) or need
recalibration on GLM activations? Check admin/knowledge and the fork's
calib tooling (tools/calib_contraction.py in the 487 tree) before phase 1.

## Upstream PR sweep (2026-09-22)
- #53906 (MERGED, in our v0.30 base): GLM-5.3 support incl. sparse-MLA fp8 KV
  storage convention (uint8 alloc viewed as e4m3) — reuse the store/spec side;
  read side is SM90/FlashInfer-only. CHERRY-PICK #55222 (open bugfix: fp8
  plan dtype + indexer prefill workspace right-sizing) when taking it.
- Net plan change: phase-1 store = upstream convention (not hand-rolled);
  phase-1 read = fork do-all sparse tile; our work = wiring + SM80 gating.
- #57128 (open): hybrid MambaManager prefix-cache corruption with MTP/EAGLE —
  we run prefix caching OFF (safe); forwarded to the thinking-off anomaly
  investigation as a related bug class.
- #57057 (open): UltraQuant 4-bit KV (FP4+UE8M0+Hadamard keys) — watch item,
  quality benchmark for our 4-bit tier; not SM80-usable as-is.

## Status 2026-09-22 (phase 1 code-complete, unvalidated)

Commits on glm53-pp4-v0300: ae31c7d173 (#55222 workspace), 13808c27e5
(hybrid hook filter), f3c5357ca3 (phase 1a kernel/store port),
a71f1204a5 (phase 1b wiring). All code-only; no GPU has run any of it.

Open questions resolved during wiring:
- GLM slot content: exactly kv_c[512], nothing else. All non-KDA layers
  are sparse-MLA v32 ("dense 0-2" = dense MLP, not dense attention), all
  NoPE (qk_rope_head_dim=0 in text_config) -> no dense-MLA TQ backend
  needed, no k_pe segment. k8v4 slot = 512 B vs 1024 B bf16 = exactly 2x.
- Scale granularity: k8v4 kv_c uses the global layer k_scale (=1.0 for
  our AWQ ckpt, no KV scales); per-token scales exist only for k_pe fp8,
  which NoPE compiles out. Phase-2 MSE carries per-token vec_norm fp16
  in-slot.
- Arena/page math: v0.30 handles packed slots natively via
  MLAAttentionSpec.state_content_bytes (fp8_ds_mla precedent); page =
  64 tok x 512 B = 32768 B, power of two, equalization-friendly.
- Fork bug avoided: its store kernels index slot*PACKED_BYTES flat and
  its forward_mqa lacks BLOCK_STRIDE_ROWS -> both would corrupt/misread
  interleaved single-arena pages. Our port is stride-aware on both paths.
- turboquant reads as NOT quantized to the layer (torch_utils
  is_quantized_kv_cache excludes it) -> q stays bf16, no fp8-attention
  side paths trigger.

Validation ladder (first GPU boot NOT yet run — coordinate with the
driver session / GPT anomaly work before touching the cards):
1. Kernel unit test, cards otherwise idle:
   .venv/bin/python -m pytest --confcutdir=tests/v1/attention \
       tests/v1/attention/test_sparse_tq_kernel.py -q
2. Boot plain (no spec decode) at 131k:
   GLM_TREE=/home/user/vllm-glm53-v0300 GLM_KV_DTYPE=turboquant_k8v4 \
       /home/user/run-glm53-pp4.sh
   Check log: "TRITON_MLA_SPARSE_TURBOQUANT" selected for the 11 MLA
   layers; KV pool tokens vs 634,124 baseline expectation.
3. Temp-0 correctness probes + BWA-MEM2 canary; then GLM_DFLASH=1
   (drafter's full-attn layers land on the native TURBOQUANT standard
   backend — SM80 support there unverified, isolate if it breaks).
4. 256k + ladder rerun + acceptance + battery.

## Validation results 2026-09-22 evening (phase 1 VALIDATED)

All rungs passed, zero Xids across 4 boots + teardowns:
1. Kernel unit test (GPU0, cards idle): PASSED — strided containment,
   NoPE layout, e5m2 roundtrip vs bf16 reference.
2. Plain 131k boot: TRITON_MLA_SPARSE_TURBOQUANT selected (sole match);
   pool 911,262 tok @ util 0.85 (6.95x). Temp-0 smokes exact; 30.7k
   dual-needle recall exact. BWA-MEM2: 27/30 vs 28/30 bf16 baseline
   (noise); 50.9 t/s plain decode, 21,993-tok generation clean.
3. DFlash rung: drafter (dense SWA + non-causal, head 128) has NO
   turboquant-capable backend -> fixed via speculative-config
   "kv_cache_dtype":"auto" (existing v0.30 hook; launcher updated).
   Smokes exact; code-gen 116.2 t/s (bf16 baseline 111); acceptance
   ~4.96 mean len vs 5.39 bf16 (small sample, watch in battery).
4. 256k DFlash boot: pool 791,640 tok / 3.02x vs bf16 634,124 / 2.42x
   at identical config = +24.8%. 200k prefill acceptance: PASSED,
   43.1s (bf16 40.3s), coherent output, zero Xids.

Why +25% and not +90%: attention block cost halved (29 vs 57 blocks per
256k request; TQ blocks hold 9216 tok vs 4096 at the same 4.72MB page
tied to the KDA state page), but the capacity sum is dominated by the
DFlash sidecar's flat 89 block-ids/request (1MiB page in every arena
block, ~22.8k-token sliding window). NEXT capacity lever = drafter
accounting/layout, not attention bytes. Also unexplained: per-rank
`available` spread at pool sizing (PP1 6.95GiB vs PP0 11.9GiB on the
131k DFlash boot).

Current server: TQ k8v4 + DFlash @ 256k on :8002 (left serving).
Prod launcher line: GLM_TREE=/home/user/vllm-glm53-v0300 GLM_DFLASH=1
GLM_LEN=262144 GLM_KV_DTYPE=turboquant_k8v4 /home/user/run-glm53-pp4.sh

## 2026-09-22 late: drafter-accounting fix handed to GPT

Brief at /home/user/reports/drafter-arena-2026-09-22/BRIEF.md. GPT owns
the box and server lifecycle for that task; this session stays off the
GPUs until its FIX-REPORT lands. Pending after: battery/Standard-14 on
the winning config (also settles the acceptance-length question),
phase 2 MSE 3-bit.

## 2026-09-22 night: drafter fix landed (GPT, e130eb1f70) + bf16 control

Pool matrix at 256k, DFlash k=7, partition 13,12,11,9, util 0.93:

| config                         | pool tokens | concurrency | core blocks/req |
|--------------------------------|------------:|------------:|----------------:|
| bf16, old layout               |     634,124 |       2.42x |            ~154 |
| TQ k8v4, old layout            |     791,640 |       3.02x |             151 |
| bf16, private drafter pool     |   1,433,053 |       5.47x |              90 |
| TQ k8v4, private drafter pool  |   1,928,026 |       7.35x |              62 |

TQ vs bf16 on the fixed layout: +34.5%. Drafter fix alone: x2.26 (bf16)
/ x2.44 (TQ). The private drafter pool caps concurrency at max_num_seqs
(713 blocks = 1 + 8*89), so 7.35x is close to that ceiling; raising
max_num_seqs would grow the PP3 reservation (713*5*1MiB = 3.48GiB now).
Both control smokes exact, zero Xids. PP1 skew explained: dense MLPs on
layers 0-2 make PP0 lighter than PP1 (partition retune lever).
Serving state restored: TQ k8v4 + DFlash @256k on :8002.

## 2026-09-22 late: full battery on prod config (TQ k8v4 + DFlash + private pool, 256k)

Run dir: llm-bench/results/glm53-pp4-v0300-tq-256k-v3.0-20260922-222047
vs bf16 baseline results/glm53-pp4-v0300-256k (same tree, pre-TQ/pre-fix).
- Practical Standard-21: 21/21 (hiveos 8/8; baseline 21/21 with hiveos 7/8).
- Decode t/s @ prompt 500/2000/4000/6000: 94.7/94.1/92.1/88.3
  (baseline 83.9/93.0/99.3/108.8). Flatter curve; single samples x3 with
  spec decode are noisy — baseline's rising curve is suspicious itself.
- Parallel t/s c1..c32: 89/108/154/212/276/296 (baseline 80/122/133/249/317/304).
  Seq 89.0 vs 75.7. Low-concurrency better, c8-c16 ~12-15% lower: plausible
  cost of TQ dequant/store at batch, or noise; worth a repeat if c8+ matters.
- Spec decode over battery: 29,057 accepted / 91,371 drafted (31.8%, mean
  accepted len ~3.2 at k=7). No bf16 battery counters exist to compare —
  workload (short answers, reasoning) differs from the code probe's 5.39.
- Parallel leg first failed: system python3 lacks aiohttp; rerun with
  /home/user/vllm-glm53-v0300/.venv/bin/python. Zero request failures.
