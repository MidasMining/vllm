# Handoff brief: DFlash drafter KV accounting dominates the GLM pool

For: GPT/codex session. From: the Claude session that did the TQ port.
Date: 2026-09-22. Host: our rig. Tree: `/home/user/vllm-glm53-v0300`
(branch `glm53-pp4-v0300`, HEAD `a71f1204a5`, pushed to Gitea
MidasMining/vllm-glm53).

## Goal

Remove (or correctly bound) the DFlash drafter sidecar's flat **89
block-ids per request** from the shared-arena capacity math, without
regressing correctness or the drafter itself. Expected payoff at 256k:
`blocks_per_request` 151 → ~62, pool 791,640 → ~1.9M tokens (~×2.4).
This is the single biggest KV-capacity lever left; it also benefits the
bf16 config identically (its baseline is 634,124 tokens / 2.42x).

## Current serving state

Server IS running: GLM-5.3-Flash, PP4, DFlash2 k=7, kv-cache-dtype
`turboquant_k8v4`, 256k, port 8002. Launch line:

    GLM_TREE=/home/user/vllm-glm53-v0300 GLM_DFLASH=1 GLM_LEN=262144 \
      GLM_KV_DTYPE=turboquant_k8v4 /home/user/run-glm53-pp4.sh

You own server lifecycle for this task (user-authorized). Rules:
- Stop with SIGTERM and wait for clean exit; NEVER hard-kill (CMP 170HX
  GSP scrub-wedge). No concurrent CUDA processes while serving.
- On any Xid: stop, record `dmesg`, report. Do not rebind/reload/reboot
  the driver yourself — driver recovery is escalated to the user (a
  separate session manages drivers). Kernel is pinned to 6.8.0-136.

## The problem, with evidence

Boot log `/home/user/glm53-tq-256k-boot.log` (also
`glm53-tq-dflash-boot.log` for the 131k variant):

    capacity inputs: num_blocks=456 blocks_per_request=151 groups=[
      (['...layers.3.self_attn.indexer.k_cache'], 6, 304128, 29),
      (['...layers.3.self_attn.indexer.tail_cache'], 3, 304128, 1),
      (['...layers.0.self_attn'], 3, 4718592, 8), ... x4 KDA groups ...,
      ([], 0, 1048576, 89)]        <-- DFlash drafter group
    GPU KV cache size: 791,640 tokens, ... 3.02x

- Concurrency = num_blocks / blocks_per_request (456/151 = 3.02 ✓).
- The drafter group demands a flat 89 block-ids/request at ANY context
  length (89 × 256-token blocks ≈ 22.8k tokens = its sliding window +
  draft margin). Attention needs only 29 blocks for a full 256k request
  (TQ; bf16 would be ~57). So the drafter is ~60% of the denominator.
- Note the drafter group appears with `layers=[]` in that log line: its
  layers live only on PP3, yet it still contributes 89 to the global
  per-request sum. Whether that empty-layers appearance is itself a bug
  in the capacity path (phantom accounting on non-PP3 ranks) or just
  cosmetic logging of a global group is the first thing to establish.
  Precedent: `_glm5_next_tensor_layout` already skips empty-layer
  groups for TENSOR EMISSION with the comment "41 phantom blocks of
  108MB draft pages shrinking the pool to 153k" — the capacity path may
  need the same treatment or a per-rank-aware version.

## Where the code is

All in `/home/user/vllm-glm53-v0300`:

- `vllm/v1/core/kv_cache_utils.py`
  - `_glm5_next_tensor_layout` (~line 1370): classifies the glm5n
    slot-sharing layout; sidecar (drafter) groups collected ~line 1417,
    empty-layer skip + padding strip immediately after.
  - glm5n packing branch (~line 1826): `per_block = mla + idx +
    sidecar_per_block`; `_region_tensors` appends a drafter region to
    EVERY arena block on the owning rank; `num_blocks = available //
    per_block`.
  - Capacity solver near the `capacity inputs` logger call (line ~1094):
    computes blocks_per_request as a SUM over groups sharing one
    block-id space. This sum semantics is correct for the slot-sharing
    core (mamba/tail parasitize MLA block-ids at disjoint ids) but is
    what over-charges the drafter.
- `vllm/v1/worker/utils.py:389` `allocate_kv_cache`: asserts ALL
  KVCacheTensors share ONE backing allocation per rank ("KV cache
  tensors must share one backing allocation"). Any "separate drafter
  allocation" design must either stay inside this contract (same arena,
  own accounting) or extend it deliberately.
- Drafter spec origin: DFlash draft attention layers (dense, sliding
  window, non-causal mix, head 128, bf16 KV — forced via
  `"kv_cache_dtype":"auto"` in the launcher's speculative-config; do
  not change that, no backend supports the drafter under turboquant).
  `VLLM_GLM5N_SIDECAR_BLOCK_SIZE=256` (launcher) reblocks it to
  256-token blocks → 1MiB page.

## Candidate designs (your call)

1. Window-aware accounting: the capacity solver charges the drafter
   `min(sliding_window_blocks, len_blocks)` — it already effectively
   does (89 is window-derived); the real issue is that those 89 ids
   come out of the SAME id budget as attention. So:
2. Separate block-id space for sidecar groups: drafter regions keep
   riding the arena (contract intact) but get their own num_blocks
   accounting; scheduler/block-table treats it as an independent pool.
   Check how block tables are built per group (`vllm/v1/worker/gpu/
   block_table.py`) before assuming this is cheap.
3. Dedicated static per-request buffer on PP3: 89MiB × max_num_seqs=8
   ≈ 712MB; PP3 has the most headroom (13.7GiB available at pool
   sizing). Cleanest math, but must reconcile with the single-backing
   assert and the runner's generic slot-mapping path (KpoolTailSpec's
   `uses_slot_mapping=False` precedent may be relevant — that fix is
   yours, commit `69dc4f43c8`).

Beware: whatever changes, the per-request block-table row sizing,
prefix-caching-off assumptions, and the PP projection of groups (empty
on 3 of 4 ranks) all have to stay consistent. The KV-ladder sidecar
reblock chain (classifier strips padding, reblocks both directions)
lives in the same code and was fragile historically.

## Validation bar (all on TQ config unless noted)

1. Boot 256k DFlash+TQ: expect blocks_per_request ≈ 62 and pool ≈
   1.8-1.9M tokens; `TRITON_MLA_SPARSE_TURBOQUANT` still selected.
2. Temp-0 smokes: 17+25→'42', capital→'Paris' (use default request or
   `include_reasoning:false`; NEVER `enable_thinking:false` — see
   `/home/user/reports/output-duplication-2026-09-22/FINDINGS.md`).
3. Spec-decode still healthy: `curl -s localhost:8002/metrics | grep
   spec_decode` — accepted/draft ratio ≈ 0.55-0.6 (baseline 1672/2954).
4. Code-gen speed ≥ ~110 t/s (TQ baseline today: 116.2 t/s).
5. 200k acceptance:
   `/home/user/vllm-glm53-v0300/.venv/bin/python
   /home/user/reports/deep-prefill-fault-2026-09-22/probe_prefill.py
   --tokens 200000 --output <out>.jsonl` — completes, coherent, ~43s.
6. Zero new Xids in dmesg across the whole session.
7. Bonus if cheap: one bf16 control boot (drop GLM_KV_DTYPE) showing
   the same structural win there.

## Secondary (optional, don't block on it)

Per-rank `available` at pool sizing is oddly skewed: 131k DFlash boot
showed PP0=11.9GiB, PP1=6.95GiB, PP2=10.96GiB, PP3=13.66GiB — PP1 has
FEWER layers than PP0 yet ~5GiB less free. If you're in the memory
accounting anyway and spot the cause, note it; the min rank sets
num_blocks, so this is worth real pool tokens too.

## Deliverables

Patch(es) on `glm53-pp4-v0300` (commit; do not push), a FIX-REPORT.md
in this directory with evidence files, and leave the server in the
state the report documents. My session stays off the GPUs until you're
done.
