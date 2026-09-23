# Deep-prefill crash: root cause, fix, and validation

2026-09-22, host our rig. Tree: `/home/user/vllm-glm53-v0300`.
Baseline commit: `4a6e6d75c8dfa5107b883f44e433eec70715b468`.

## Outcome

The patched server completed an exact 200,000-token prompt and generated 32
tokens in 40.276 seconds at the original utilization 0.93, PP4 partition
13,12,11,9, max length 262144, and DFlash2 configuration. CUDA launches were
asynchronous; temporary diagnostics were removed. No new Xids occurred during
startup, the long request, or subsequent smoke requests. The server remains
running on port 8002. Source and regression test changes are uncommitted.

## Root cause

`KpoolTailSpec` allocates one block-table column per request, but inherited
`uses_slot_mapping=True`. The V2 runner's generic slot-mapping kernel therefore
looked up `table[request, position // 4]`. This leaves the request's row as soon
as position reaches 4 and eventually leaves mapped memory entirely.

The existing `KpoolTailMetadataBuilder` subsequently replaces the generic
results with the correct circular mapping, but the illegal read has already
happened. Earlier prompts can survive because their bad addresses are still
mapped. Allocation placement can shift the visible crash threshold without
there being a physical VRAM ceiling. This is not an int32-overflow defect.

The fix overrides `KpoolTailSpec.uses_slot_mapping` to return False. The generic
worker produces placeholder slots without indexing by growing position; the
dedicated builder continues to supply `own_block * kpool + position % kpool`.
The existing tail-cache consumers use that builder's metadata.

## Isolation evidence

1. Clean driver reload and exact 200k-token reproduction with synchronization
   at KDA entry and after convolution, state gathering, chunk computation and
   state scatter. Fault reproduced at 98,136 computed + 4,089 scheduled tokens.
   All checked operations in the preceding chunk passed. Failure at layer-0
   KDA entry placed the source before that chunk's KDA computation.
2. Another clean reload with `CUDA_LAUNCH_BLOCKING=1` identified
   `_compute_slot_mappings_kernel` in `vllm/v1/worker/gpu/block_table.py:204`,
   called by `model_runner.prepare_attn`. Kernel journal: 16:14:54,
   FAULT_PDE ACCESS_TYPE_VIRT_READ. This run hung after fault; its probe is not
   a successful completion.
3. Source inspection established the mismatch between one-column capacity,
   the inherited slot-mapping flag, and position-based generic lookup.

Evidence: `diagnostic-serve.log`, `diagnostic-probe.jsonl`,
`blocking-serve.log`, `blocking-xid.txt`, `blocking-probe.jsonl`.

## Patch and tests

- Production fix: six added lines in `vllm/v1/kv_cache_interface.py`.
- Regression: `tests/v1/worker/test_gpu_block_table.py`. Two requests with
  permuted batch order, one-column tables, positions 98135, 98136, 102224,
  199999, 200000, 262143, and padding. Checks generic bypass and exact circular
  slots. An initial assertion safely rejects the unfixed flag before launching
  an unsafe GPU read.
- On original source: regression fails (`regression-before.txt`).
- Fixed GPU block-table suite: 10 passed (`regression-after.txt`).
- Existing tail mapping suite: 12 passed (`tail-mapping-tests.txt`).
- `git diff --check`: passed.
- Full-model 200k completion: `fixed-probe-200k.jsonl`, `fixed-serve.log`.
- Zero post-fix Xids: `fixed-xids.txt` (empty).

Commands from the repository:

```sh
.venv/bin/python -m pytest --confcutdir=tests/v1/worker tests/v1/worker/test_gpu_block_table.py -q
.venv/bin/python -m pytest --confcutdir=tests/v1/attention tests/v1/attention/test_kpool_tail_slot_mapping.py -q
```

`fix.patch` contains the final source/test diff. Temporary KDA instrumentation
is preserved separately as `diagnostic.patch` and was removed from the tree.
pytest was installed through uv; a stalled ruff download was cancelled, so
ruff was not run.

## Output-quality limitation and scope

After the 200k request, default-mode short requests correctly answered `4`
and `Paris`. With `enable_thinking=False`, the arithmetic prompt asking for
17+25 reproducibly returned `17 + 25 = 4242`. That is not a clean correctness
pass. Its origin (model/template/speculative decoding or another issue) has
not been isolated; do not attribute it to speculation based on this evidence.
See `fixed-smoke.json`, `quality-probe.jsonl`, and `quality_probe.py`.

The long-context test establishes successful processing and generation, not
long-context retrieval accuracy or comprehensive model correctness.

This fix addresses the reproduced deep-prefill read fault. Older load-phase
REGION_VIOLATION write faults and the separate NCCL/P2P incidents remain
separate investigations. The initial review's KDA suspicion was superseded
by the synchronous launch evidence.
