# GLM-5.3-Flash on four CMP 170HX mining cards

A ~180 GB mixture-of-experts model serving up to its full 1M-token context
on four ex-mining GPUs over PCIe Gen2 with no NVLink, at 90–165 tokens/s on
short contexts and about 50 t/s deep in a 1M one. Measured
2026-09-22 on branch `glm53-pp4-v0300` (vLLM v0.30.0 fork). The same
content as a styled page is in [`cookbook.html`](cookbook.html).

## At a glance

| Metric | Value |
| --- | --- |
| KV cache at 256k context | **1,928,026 tokens** (7.35 full-length requests) |
| Code generation, single stream | ~165 t/s |
| Single-user decode, mixed chat (500–6k token prompts) | 89–95 t/s |
| Aggregate throughput, 32 concurrent | ~296 t/s |
| Max context per request | **1,048,576 tokens** (native); 3/3 needle recall at 1.04M |
| 200k-token prefill | ~43 s (~4,650 tok/s); ~4.6 min for 1M |
| Practical debugging suite / BWA-MEM2 review case | 21/21 / 27 of 30 |

## Hardware and model

| | |
| --- | --- |
| GPUs | 4× NVIDIA CMP 170HX (GA100 / SM80), 64 GB each, unlocked driver 610.43.03 |
| Interconnect | PCIe Gen2 x16 per card, no NVLink, NCCL P2P disabled |
| Model | GLM-5.3-Flash AWQ W4A16, 178 GB. 45 layers: 11 sparse-MLA (indexer top-2048), 34 KDA linear attention; 288 experts, 8 active |
| Drafter | GLM-5.3-Flash-DFlash2 (2.2 GB), 7 speculative tokens |
| Parallelism | Pipeline parallel PP4, layer split 13 / 12 / 11 / 9 |
| Software | vLLM v0.30.0 + this branch, CUDA 13.3, Linux 6.8 |

## Context and KV cache

The model's native limit is 1,048,576 tokens and the rig runs it at the full
1M (see Long-context results). The benchmarks were taken at a 262,144 limit.

| KV configuration (256k, DFlash on) | Cache tokens | Full-length requests |
| --- | ---: | ---: |
| bf16 KV, original layout | 634,124 | 2.42 |
| TurboQuant fp8 KV, original layout | 791,640 | 3.02 |
| bf16 KV, private drafter pool | 1,433,053 | 5.47 |
| **TurboQuant fp8 KV + private drafter pool** | **1,928,026** | **7.35** |
| Same, with the limit raised to 1M | 2,639,272 | 2.52 at 1M |

- **TurboQuant fp8 KV** (`turboquant_k8v4`): the sparse-attention layers
  store their 512-wide latent as one fp8 byte per value instead of two
  bf16 bytes, which halves attention cache pages. SM80 stores e5m2 and
  decodes with a bitcast. Decode is no slower.
- **Private drafter pool**: the DFlash drafter only needs a ~22.8k-token
  sliding window, but it had been using 89 block IDs per request from the
  main pool. It now has its own fixed 3.5 GB pool on the last card, which
  cut main-pool demand per 256k request from 151 blocks to 62.

The drafter pool is sized for `--max-num-seqs 8`, so 7.35 is close to the
ceiling for this config.

## Benchmarks

Temperature 0 against the live server. The baseline is the same model and
branch earlier the same day, with bf16 KV and without the drafter-pool fix.

| Concurrent | Current t/s | Baseline t/s |
| ---: | ---: | ---: |
| 1 | 89 | 80 |
| 2 | 108 | 122 |
| 4 | 154 | 133 |
| 8 | 212 | 249 |
| 16 | 276 | 317 |
| 32 | 296 | 304 |

One run per level with speculative decoding on, so expect ±10% run to
run. Concurrency 8–16 came in 13–15% lower than baseline, which is either
a real cost of the fp8 dequant at larger batches or noise.

| Prompt tokens | Decode t/s (current) | Decode t/s (baseline) | Time to first token |
| ---: | ---: | ---: | ---: |
| 500 | 94.7 | 83.9 | 0.89 s |
| 2,000 | 94.1 | 93.0 | 1.51 s |
| 4,000 | 92.1 | 99.3 | 2.53 s |
| 6,000 | 88.3 | 108.8 | 2.88 s |

Other results: 200k prefill + 32 tokens in 43.1 s with a correct answer;
exact two-needle recall at 30.7k tokens; 24/24 concurrent smoke requests
correct; plain decode without the drafter 50.9 t/s; zero GPU faults (Xid)
across every boot and test. Speculative acceptance is about 5 tokens per
step on code and averaged about 3.2 of 7 on the mixed battery.

## Long-context results (1M boot)

Launched with `--max-model-len 1048576`. Temperature 0, reasoning on.

Needle recall, three facts at 10%, 50% and 90% depth:

| Prompt tokens | Recall | Time to answer |
| ---: | :---: | ---: |
| 65,490 | 3/3 | 17.8 s |
| 131,025 | 3/3 | 30.1 s |
| 262,098 | 3/3 | 59.8 s |
| 524,242 | 3/3 | 125.2 s |
| 786,386 | 3/3 | 200.4 s |
| **1,039,954** | **3/3** | **281.2 s** |

Speed and reasoning at depth (mining-pool source files scattered through
log filler; trace a share across files, answer four cross-file questions):

| Context | Time to first token | Decode t/s | Cross-file trace | Cross-reference |
| ---: | ---: | ---: | ---: | ---: |
| 131k | 28.6 s | 56 | 6/6 | 4/4 |
| 524k | 122.9 s | 54 | 6/6 | 4/4 |
| 1.04M | 278.1 s | 48 | 6/6 | 4/4 |

Growing conversation: an 11-turn session grew by 45k tokens per turn to
496k. Every turn recalled both the newest fact and one from the first turn,
with no looping.

> **Every turn re-reads the whole conversation.** Prefix caching is off in
> this configuration, so a follow-up turn waits as long as the first: about
> 1 minute at 256k, 2 at 512k, 4.6 at 1M. The 11-turn session spent 11.3
> minutes in prefill, where reading only the new text would take about 2.
> Prefix caching for this layout is in progress.

## Launch recipe

```bash
# --- environment ---
export CUDA_DEVICE_ORDER=PCI_BUS_ID
export CUDA_VISIBLE_DEVICES=0,1,2,3
export NCCL_P2P_DISABLE=1              # no usable P2P on these cards
export NCCL_IB_DISABLE=1
export VLLM_WORKER_MULTIPROC_METHOD=spawn
export VLLM_PP_LAYER_PARTITION=13,12,11,9   # layers per card; card 0 caps at 13
export VLLM_GLM5N_SIDECAR_BLOCK_SIZE=256    # drafter KV block size
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:False
export HF_HUB_OFFLINE=1

# --- server ---
python -m vllm.entrypoints.openai.api_server \
  --model /models/GLM-5.3-Flash-AWQ-W4A16 \
  --served-model-name glm53-flash \
  --pipeline-parallel-size 4 \
  --max-model-len 262144 \
  --max-num-seqs 8 \
  --max-num-batched-tokens 4096 \
  --gpu-memory-utilization 0.93 \
  --kv-cache-dtype turboquant_k8v4 \
  --no-enable-prefix-caching \
  --disable-custom-all-reduce \
  --speculative-config '{"method":"dflash","model":"/models/GLM-5.3-Flash-DFlash2","num_speculative_tokens":7,"kv_cache_dtype":"auto"}' \
  --enable-auto-tool-choice --tool-call-parser glm47 \
  --reasoning-parser glm45 \
  --host 0.0.0.0 --port 8002
```

Cold start takes about 4 minutes; on the first boot, Triton kernel
compilation adds several more.

| Setting | Why |
| --- | --- |
| `--max-model-len` | 262144 is the benchmarked setting. 1048576 enables the full 1M context and still serves short requests; its throughput under many concurrent mid-size requests hasn't been benchmarked. |
| `--kv-cache-dtype turboquant_k8v4` | fp8 attention cache. Remove for bf16 KV at about 26% fewer cache tokens. |
| `"kv_cache_dtype":"auto"` in the drafter config | Required: no backend runs the drafter's sliding-window attention on a TurboQuant cache. |
| `--gpu-memory-utilization 0.93` | Tested ceiling; 0.95 caused faults. |
| Partition `13,12,11,9` | Card 0 faults loading more than 13 MoE layers. Layers 0–2 have dense MLPs, so card 1 ends up the tightest. |
| `--disable-custom-all-reduce` | Its BAR1 mappings triggered copy-engine faults on these cards. |
| `--no-enable-prefix-caching` | Required by the private drafter pool and the hybrid KV layout. |
| `--max-num-seqs 8` | Sets the drafter pool size (8 × 89 blocks). |

## Using it

OpenAI-compatible API. The model always reasons; the reasoning comes back
in a separate field.

```bash
curl http://HOST:8002/v1/chat/completions -H 'Content-Type: application/json' -d '{
  "model": "glm53-flash",
  "messages": [{"role": "user", "content": "What is 17 plus 25?"}],
  "temperature": 0,
  "max_tokens": 4096,
  "include_reasoning": false
}'
```

> **Don't send `enable_thinking: false`.** The chat template ignores it
> but the parser stops separating reasoning from the answer, giving
> outputs like `17 + 25 = 4242`. Use `"include_reasoning": false` to hide
> reasoning (it is still generated and billed).

Leave room for reasoning in `max_tokens`: hard questions can reason for
20k+ tokens. Tool calling works through the `glm47` parser.

## Operating notes for these cards

- **Stop with SIGTERM, never SIGKILL.** Killing a worker mid VRAM scrub can
  wedge the GPU firmware; only a reboot recovers it.
- **Wait before restarting.** Workers can hold VRAM for minutes after the
  API server exits. Relaunch only when `nvidia-smi` shows no compute
  processes and the port is free.
- **One CUDA workload at a time** while serving.
- **Pin the kernel.** The unlocked driver is built for one kernel version.
- **After an Xid 31**, a full driver-module reload usually clears it;
  `nvidia-smi --gpu-reset` does not.

## Credits

Built on promisezackr's glm53-flash-170hx-pp8 (PP8 recipe and patches) and
the TurboQuant MLA fork (sparse TQ kernels). KV and drafter-pool fixes
made in-house and validated 2026-09-22. Numbers are from a single rig and
a single day of runs.

## Files in this directory

- `cookbook.html`: this cookbook as a styled page
- `run-glm53-pp4.sh`: the launcher used on the rig (`GLM_DFLASH=1 GLM_LEN=262144 GLM_KV_DTYPE=turboquant_k8v4`)
- `bench/`: raw decode, parallel and practical-suite JSON for the numbers above; `bench/long-context/` has the 1M needle ladder and marathon battery scripts and results
- `reports/`: engineering write-ups: TurboQuant port design and validation, deep-prefill Xid 31 fix, drafter-pool fix (brief and report), the `enable_thinking` duplication finding, BWA-MEM2 scoring
