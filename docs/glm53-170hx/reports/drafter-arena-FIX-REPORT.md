# DFlash drafter arena fix

Host: `our rig`. Tree: `/home/user/vllm-glm53-v0300`.
Branch: `glm53-pp4-v0300`. Base: `a71f1204a5fbdbf60ca0e395647718fd3e4aab88`.
Investigation: 2026-09-22, America/Chicago. Commit recorded in `commit.txt`.

## Result

The 256k TQ/DFlash server now reports **1,928,026 cache tokens / 7.35x
concurrency**, versus **791,640 / 3.02x** before: approximately **2.44x**.
Core blocks per request are **62 instead of 151**, with the same 456 core
blocks available. The drafter remains enabled with seven speculative tokens
and unquantized bf16 KV.

| Check | Result |
|---|---|
| Backend | `TRITON_MLA_SPARSE_TURBOQUANT` |
| Arithmetic / capital | Exactly `42` / `Paris` |
| Concurrent requests | 24/24 correct, three rounds of eight |
| Matched raw code generation, 2,048 tokens | 159.41 t/s before, 165.23 t/s after, including prefill |
| Same run, decode interval | 161.99 t/s before, 166.93 t/s after |
| Prescribed 200k probe | 200,000 prompt + 32 completion tokens, 43.15 seconds |
| Additional chat-formatted 200k probe | Exactly `7`, 41.92 seconds, normal stop |
| DFlash final metrics | 3,511 accepted / 6,685 drafted tokens, 52.52% |
| New Xids | Zero in kernel journal since 20:10 local, covering the investigation |
| Final endpoint | Healthy, HTTP 200 on port 8002 |

The acceptance ratio is workload-dependent. This session's mix includes a
long reasoning request and short concurrent smokes; it is not the brief's
1672/2954 workload. Speculation is active and the matched raw-code speed
exceeds the requested 110 t/s floor. The exploratory chat code request used
all 2,048 tokens for reasoning: 86.95 t/s before and 81.04 after, with no final
code before the length limit. Those single-sample measurements are retained
in `baseline.jsonl` and `fixed-tq.jsonl`; the raw-completion comparison is the
reported code-generation benchmark.

The prescribed repetitive raw prefill probe repeated “Write the number seven”
after producing 7. It establishes successful long-context execution, not a
strong answer-quality result. The additional properly chat-formatted probe
had exactly 200,000 prompt tokens and returned `</think>7`, with stop rather
than length termination. No `enable_thinking=false` requests were used.

## Diagnosis

The empty drafter `layer_names=[]` group on PP0 is a global scheduler
placeholder, not disposable capacity overhead. Before the fix, its manager
really took 89 IDs per full-window request from the same BlockPool as target
attention, KDA and the kpool tail. Skipping the placeholder in the capacity
logger would have advertised capacity without providing it.

The target groups intentionally share physical regions at disjoint IDs. The
drafter already has disjoint tensor regions on PP3, so it can safely use its
own ID namespace once allocation, admission and worker views agree.

## Implementation

* Mark eligible DFlash sliding-window groups before PP projection. Preserve
  their independent capacity on every rank, including empty placeholders.
* Reserve `1 + max_num_seqs * window_bound` blocks per private group. Here that
  is `1 + 8 * 89 = 713`, including its null block.
* Keep one backing allocation per rank. Reserve the private regions first in
  the sizing math, and append them after the target regions in the arena.
  Resizing the shared pool across PP ranks preserves the fixed reservation.
* Give the scheduler a separate BlockPool per private group. Admission checks
  each pool before allocation; a full private pool defers admission. Existing
  sliding-window recycling and pool-owned free operations remain in use.
* Give worker tensors their actual independent block count. Generic dense
  slot mapping and existing per-request table row sizing remain applicable.
* Exclude private bf16 cache from core block zeroing, including new-block ID
  collection. The same numeric ID can simultaneously name unrelated live
  target and drafter pages. Warmup also allocates IDs per pool and checks
  pool-specific bounds.
* Apply explicit block overrides before rejecting a zero memory budget:
  profiling deliberately creates an eight-block dummy arena with budget zero.

There are **five** drafter layers sharing the group's block table. Thus the
physical reservation is `713 * 5 * 1 MiB = 3.481 GiB` on PP3, not 713 MiB total.
PP0–PP2 reserve no drafter storage. PP1 remains the limiting core-pool rank.
The resulting PP3 single backing allocation is 8,318,894,080 bytes.

The optimization is restricted to the recognized GLM layout, DFlash,
unquantized sliding-window draft KV, prefix caching disabled, and no KV
transfer connector. Other configurations keep the existing shared-pool path.
The previous KpoolTailSpec deep-prefill fix is preserved.

## Tests and checks

* `unit-core.txt`: 135 passed; four logging tests initially lacked the
  `caplog_vllm` fixture because the run used `--confcutdir=tests/v1/core`.
* `unit-logging.txt`: those four tests passed after loading the original
  logging fixtures through the report-local `logfixtures.py` plugin.
* `unit-worker.txt`: 43 passed, including actual GPU zeroing containment and
  independent warmup-ID checks. GPU unit tests ran only while serving was off.
* `unit-private-final.txt`: five focused cases passed, covering PP projection,
  arena resizing, admission/recycling/free, profiling's zero-budget override,
  and retaining the shared pool for quantized drafts.
* `lint-audit.json`: zero introduced Ruff diagnostics. The base already has
  five diagnostics in `kv_cache_utils.py` and 69 in the GPU model runner;
  those remain. All changed Python files compile; `git diff --check` passes.

Main test commands:

```sh
.venv/bin/python -m pytest --confcutdir=tests/v1/core \
  tests/v1/core/test_kv_cache_utils.py \
  tests/v1/core/test_swa_inflight_window_free.py \
  tests/v1/core/test_deferred_block_free.py -q
.venv/bin/python -m pytest --confcutdir=tests/v1/worker \
  tests/v1/worker/test_gpu_warmup_blocks.py \
  tests/v1/worker/test_kv_block_zeroer.py -q
```

The final extra guards were checked with the focused selector
`private_drafter or profiling_override or quantized_drafter`. A separate bf16
control boot and a broad model-quality benchmark were not run. No claim of
bit-identical generation or exhaustive model-quality equivalence is made.

## Evidence and final state

All files below are in this report directory:

* `fixed-tq-boot-v2.log`: successful boot, capacity and backend evidence.
* `baseline-codegen.json`, `fixed-tq-codegen.json`: identical raw benchmark
  request definitions, timings and all streamed chunks.
* `fixed-tq-200k.jsonl`: prescribed probe's streamed output and usage.
* `fixed-tq-chat-200k.json`: additional exact-token probe, answer and timings.
* `concurrent-smoke.jsonl`: all 24 independent requests and responses.
* `metrics-final.txt`, `kernel-journal.txt`, `dmesg-before.txt`,
  `dmesg-after.txt`: final speculative counters and kernel evidence.
* `fix.patch`, `commit.txt`: reviewable implementation and committed identity.

Server left **running** on port 8002: API PID 244096, EngineCore 244654,
workers 244911–244914. Launch:

```sh
GLM_TREE=/home/user/vllm-glm53-v0300 GLM_DFLASH=1 GLM_LEN=262144 \
  GLM_KV_DTYPE=turboquant_k8v4 /home/user/run-glm53-pp4.sh
```

The final source adds a guard against quantized draft caches after this live
boot. It does not change the active bf16 drafter path; the guard is CPU-tested
and takes effect on subsequent launches. An earlier test boot was stopped
with SIGTERM to correct profiling override ordering. All remaining processes
were allowed to exit after SIGTERM; no SIGKILL, driver reload/rebind or reboot
was performed. No patch was pushed.

## Secondary finding: PP1 memory skew

The checkpoint's first three MLPs are dense. With partition 13/12/11/9:

| Rank | Dense MLPs | MoE MLPs | Serialized target layer weights |
|---|---:|---:|---:|
| PP0 | 3 | 10 | 39.42 GiB |
| PP1 | 0 | 12 | 45.38 GiB |
| PP2 | 0 | 11 | 41.59 GiB |
| PP3 | 0 | 9 | 34.04 GiB |

These are safetensors-header byte counts from the checkpoint index, excluding
embeddings/head, drafter, runtime repacking and allocator overhead
(`weight-inventory.json`). PP1 has about 5.96 GiB more serialized layer weights
than PP0 despite having fewer layers. The baseline runtime's consumed
weights/non-torch figures are 47.18 versus 42.29 GiB. This explains the bulk
of the skew; it does not indicate a five-GiB KV-accounting leak. No partition
changes were made.
