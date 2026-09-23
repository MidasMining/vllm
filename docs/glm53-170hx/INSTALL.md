# Installing GLM-5.3-Flash on four CMP 170HX: exact replication

This reproduces the rig behind [README.md](README.md) step by step, with
the exact versions it runs. Each step ends with a check; don't move on
until it passes. Everything here was recorded from the working rig on
2026-09-23 unless a step is marked **GAP**.

## What you are building

| Layer | Exact version on the rig |
| --- | --- |
| GPUs | 4× NVIDIA CMP 170HX (GA100, PCI ID `10de:20c2`), unlocked to 64 GB each |
| Host | 96 threads, 125 GB RAM, NVMe (the model alone is 180 GB) |
| OS | Ubuntu 24.04 LTS, glibc 2.39, gcc 13.3 |
| Kernel | `6.8.0-136-generic`, pinned |
| Driver | `nvidia-open` 610.43.03 with cmpunlocker patches, module at `/lib/modules/6.8.0-136-generic/updates/cmpunlocker/nvidia.ko` |
| CUDA toolkit | 13.3.1 (`cuda-toolkit-13-3` from NVIDIA's apt repo), `/usr/local/cuda-13.3` |
| Python | 3.12.3 (system), venv made by `uv` 0.12.0 |
| vLLM | v0.30.0 + branch `glm53-pp4-v0300` (reports as `0.30.1.dev1+g3d2c1a121.cu133`) |
| Key wheels | torch 2.13.0+cu132, triton 3.7.1, flashinfer-python 0.6.18.post1, transformers 5.17.0, compressed-tensors 0.17.0 |
| Model | `wtdcode/GLM-5.3-Flash-AWQ-W4A16`, 178 GB, HF revision `abd7b07719111f137e1de8a0c1b7e01c11b74d1a` |
| Drafter | `incoai/GLM-5.3-Flash-DFlash2`, 2.2 GB |

## 1. Operating system and kernel pin

Install Ubuntu 24.04 and boot kernel `6.8.0-136-generic`. The unlocked
driver is built for exactly this kernel. It is not registered with DKMS, so
any kernel update boots with **zero GPUs** even though `lspci` still shows
all four.

Pin it three ways. Holding the metapackage alone isn't enough: unattended
upgrades still pulled in a newer versioned kernel on the rig.

```bash
sudo apt-mark hold linux-image-generic linux-headers-generic linux-generic \
  linux-image-6.8.0-136-generic linux-headers-6.8.0-136-generic \
  linux-modules-6.8.0-136-generic
```

In `/etc/apt/apt.conf.d/50unattended-upgrades`, add to
`Unattended-Upgrade::Package-Blacklist`:

```text
"linux-image-";
"linux-headers-";
"linux-modules-";
"linux-tools-";
```

Make 6.8.0-136 the GRUB default. Find its menu IDs with
`grep -E "submenu|menuentry" /boot/grub/grub.cfg`, then set in
`/etc/default/grub`:

```text
GRUB_DEFAULT="gnulinux-advanced-<UUID>>gnulinux-6.8.0-136-generic-advanced-<UUID>"
```

and run `sudo update-grub`. A `GRUB_DEFAULT` that matches no entry silently
falls back to the newest kernel, so check `grub.cfg` for the resulting
`set default=` line.

**Check:** `uname -r` prints `6.8.0-136-generic`.

## 2. Driver and unlock

**GAP:** the rig's driver is a hand-built `nvidia-open` 610.43.03 carrying
cmpunlocker patches (GSP firmware mode, 64 GB VRAM, restored compute).
Its build steps are maintained separately and are not recorded here yet.
The public starting point is the cmpunlocker project
([fulracoco/cmpunlocker](https://github.com/fulracoco/cmpunlocker) and its
forks by kinako404 and abobasixseven). Its README targets `nvidia-open`
580.159.04, not 610.43.03.

Whatever route you take, the installed state must match this:

```bash
modinfo nvidia | grep -E "^version|^filename"
# version:  610.43.03
# filename: /lib/modules/6.8.0-136-generic/updates/cmpunlocker/nvidia.ko
nvidia-smi --query-gpu=name,memory.total,pcie.link.gen.max --format=csv
# NVIDIA CMP 170HX, 65536 MiB, 2   (four rows)
```

Hold the driver packages in apt (`nvidia-dkms-open`, `nvidia-driver-open`,
`nvidia-open`) so an update can't replace the patched build.

**Check:** `nvidia-smi -L` lists four GPUs and each reports 65536 MiB.

## 3. Toolkit and system packages

```bash
# NVIDIA CUDA apt repo for Ubuntu 24.04 must be configured first
sudo apt install cuda-toolkit-13-3 build-essential git ninja-build
curl -LsSf https://astral.sh/uv/install.sh | sh     # provides uv
```

- `git` is required: the vLLM build reads its version from git and fails
  or mislabels itself without it.
- `ninja-build` (system ninja) is required at serve time: FlashInfer
  compiles kernels on first launch. Without it one pipeline worker crashes
  and the others wait forever, which looks like a very slow load.

**Check:** `/usr/local/cuda-13.3/bin/nvcc --version` reports 13.3, and
`uv --version` works.

## 4. Build vLLM from the branch

```bash
cd ~
git clone -b glm53-pp4-v0300 --single-branch https://github.com/MidasMining/vllm.git vllm-glm53-v0300
cd vllm-glm53-v0300
git config user.name you && git config user.email you@example.com   # optional

uv venv -q --python 3.12
VLLM_TARGET_DEVICE=cuda TORCH_CUDA_ARCH_LIST="8.0" \
  CUDA_HOME=/usr/local/cuda-13.3 PATH=/usr/local/cuda-13.3/bin:$PATH \
  MAX_JOBS=32 NVCC_THREADS=2 \
  uv pip install -e . --torch-backend=auto 2>&1 | tee /tmp/vllm-build.log
```

- `TORCH_CUDA_ARCH_LIST="8.0"` builds only for SM80, which cuts build time
  a lot. The cards are GA100.
- `MAX_JOBS=32 NVCC_THREADS=2` fits 125 GB of RAM. Higher parallelism ran
  out of memory on this host.
- The build fetches its pinned dependencies (FlashMLA `0eee43b`,
  vllm-project flash-attention `506341a`, CUTLASS, DeepGEMM) from GitHub
  by commit, so the machine needs internet access for this step.
- Compiled extensions are host-specific. Copying a built tree to another
  machine fails with errors like `GLIBC_2.38 not found`. Rebuild in place:
  `rm -rf build && CUDA_HOME=/usr/local/cuda-13.3
  PATH=$PWD/.venv/bin:/usr/local/cuda-13.3/bin:$PATH TORCH_CUDA_ARCH_LIST=8.0
  MAX_JOBS=64 NVCC_THREADS=4 .venv/bin/python setup.py build_ext --inplace`.

**Check:**

```bash
.venv/bin/python -c "import vllm, torch; print(vllm.__version__, torch.__version__)"
# 0.30.1.dev1+g3d2c1a121.cu133 2.13.0+cu132   (version suffix follows your git HEAD)
.venv/bin/python -m pytest --confcutdir=tests/v1/attention \
  tests/v1/attention/test_sparse_tq_kernel.py -q
# 1 passed   (TurboQuant kernel on a real GPU; run with the GPUs otherwise idle)
```

## 5. Download the model and drafter

About 181 GB total. Put both in one directory; the launcher defaults to
`/mnt/ssd/models` and accepts `GLM_MODEL_DIR` to change it.

```bash
M=/mnt/ssd/models
.venv/bin/hf download wtdcode/GLM-5.3-Flash-AWQ-W4A16 --revision abd7b07719111f137e1de8a0c1b7e01c11b74d1a \
  --local-dir $M/GLM-5.3-Flash-AWQ-W4A16 --max-workers 4
.venv/bin/hf download incoai/GLM-5.3-Flash-DFlash2 --local-dir $M/GLM-5.3-Flash-DFlash2
```

The checkpoint is [wtdcode/GLM-5.3-Flash-AWQ-W4A16](https://huggingface.co/wtdcode/GLM-5.3-Flash-AWQ-W4A16):
AWQ W4A16 (group 128, compressed-tensors `pack-quantized`) of
`zai-org/GLM-5.3-Flash`, calibrated from the official FP8 release, with
attention, shared experts, dense MLPs and the MTP layer kept in BF16. The
`--revision` pins the exact snapshot we measured; a different quantization
changes memory fit and quality.

The DFlash2 drafter is licensed CC BY-NC-ND 4.0 (non-commercial).

**Check:** `du -sh $M/GLM-5.3-Flash-AWQ-W4A16` is about 178 GB with nine
`model-0000N-of-00009.safetensors` shards; the drafter is about 2.2 GB.

## 6. Launch

The launcher is [`run-glm53-pp4.sh`](run-glm53-pp4.sh). Copy it anywhere.
It sets every required environment variable and refuses to start while
GPU processes from a previous server are still running.

```bash
# 1M context (current production)
GLM_TREE=$HOME/vllm-glm53-v0300 GLM_DFLASH=1 GLM_LEN=1048576 \
  GLM_KV_DTYPE=turboquant_k8v4 ./run-glm53-pp4.sh > glm53.log 2>&1 &

# 256k context (the benchmarked configuration)
GLM_TREE=$HOME/vllm-glm53-v0300 GLM_DFLASH=1 GLM_LEN=262144 \
  GLM_KV_DTYPE=turboquant_k8v4 ./run-glm53-pp4.sh > glm53.log 2>&1 &
```

Add `GLM_MODEL_DIR=/your/models` or `GLM_PORT=...` if yours differ. What
the switches set is explained in the README's "Why these settings" table.

The first launch takes 10–20 minutes while Triton and FlashInfer compile
kernels. Later launches take about 4 minutes.

**Check:** the log shows all four of these lines:

```text
Using TRITON_MLA_SPARSE_TURBOQUANT attention backend
glm5n private drafter pool: layers=[...45..49...] blocks=713 blocks_per_request=89
GPU KV cache size: 2,639,272 tokens, Maximum concurrency for 1,048,576 tokens per request: 2.52x
Application startup complete.
```

At 256k the cache line reads `1,928,026 tokens ... 7.35x`. Small
differences are normal if your cards or partition differ, but a much
smaller pool means something is misconfigured.

## 7. Verify

```bash
curl -s localhost:8002/v1/chat/completions -H 'Content-Type: application/json' -d '{
  "model": "glm53-flash",
  "messages": [{"role": "user", "content": "What is 17 plus 25? Give only the number."}],
  "temperature": 0, "max_tokens": 512, "include_reasoning": false}'
# content: "42"
```

Then repeat the published numbers with the scripts in
[`bench/long-context/`](bench/long-context/):

```bash
# three-needle ladder, 64k to 1.04M (needs the 1M launch); every rung should be 3/3
.venv/bin/python docs/glm53-170hx/bench/long-context/needle_ladder.py \
  65536,131072,262144,524288,786432,1040000 needle.jsonl
# speed at depth, cross-file reasoning, 11-turn growing session
.venv/bin/python docs/glm53-170hx/bench/long-context/marathon_bench.py \
  marathon.jsonl speed,xfile,soak
```

Both scripts load the tokenizer from `$GLM_MODEL_DIR` (default
`/mnt/ssd/models`). `marathon_bench.py` also imports the mining-pool test
files from our llm-bench repo; point `LLM_BENCH_DIR` at a checkout of it. The short-context battery (decode,
parallel, practical suite) comes from our llm-bench repo; its raw results
are in [`bench/`](bench/).

## 8. Operating it

- **Stop:** `kill -TERM <api_server pid>`. Never SIGKILL: killing a worker
  while the driver scrubs its VRAM can wedge the GPU firmware, and only a
  reboot recovers that.
- **Restart:** wait until `nvidia-smi --query-compute-apps=pid
  --format=csv,noheader` prints nothing and the port is free. This can
  take minutes after the API server exits. The launcher's preflight
  enforces this.
- **Don't share the GPUs:** run no other CUDA program while serving, not
  even a quick test.
- **Requests:** never send `chat_template_kwargs: {"enable_thinking":
  false}`. It produces merged output like `17 + 25 = 4242`. Use
  `"include_reasoning": false` to hide reasoning instead.
- **After a reboot:** check `uname -r`, then `nvidia-smi -L` (four GPUs,
  65536 MiB each) before launching.

## Troubleshooting

| Symptom | Cause | Fix |
| --- | --- | --- |
| `nvidia-smi` shows 0 GPUs after a reboot, `lspci -d 10de:` shows 4 | Booted a newer kernel that has no patched module | Boot 6.8.0-136 (`grub-reboot`), then finish the kernel pin in step 1 |
| Load seems to hang forever, CPU idle | One pipeline worker crashed, usually FlashInfer JIT without system ninja | `grep ERROR` in the log; install `ninja-build` |
| `GLIBC_2.38 not found` | Extensions built on a different host | Rebuild in place (step 4) |
| `No valid attention backend found ... kv_cache_dtype=turboquant_k8v4 ... sliding window` | Drafter asked to use the TurboQuant cache | Keep `"kv_cache_dtype":"auto"` in the speculative config (the launcher does) |
| `Private drafter pools require prefix caching off` | Prefix caching enabled | Keep `--no-enable-prefix-caching` |
| `Errno 98 Address already in use` | Previous server still exiting | Wait for it to drain (section 8) |
| Kernel log shows `Xid 31` and the engine dies | GPU memory fault | Stop the server. A full driver-module reload usually clears it; `nvidia-smi --gpu-reset` does not. If Xid 154 also fired, reboot |
| Faults when loading layers on the first card | More than 13 MoE layers on GPU 0 | Keep the `13,12,11,9` partition, util ≤ 0.93 |
| Output like `4242`, reasoning mixed into the answer | `enable_thinking: false` in the request | Remove it; use `include_reasoning: false` |
