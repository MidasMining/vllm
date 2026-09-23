# BWA-MEM2 case — GLM-5.3-Flash (AWQ-W4A16, PP4 plain + TQ k8v4 KV, 4x CMP 170HX .57)

Date: 2026-09-22. temp 0, max_tokens 32000, kv-cache-dtype=turboquant_k8v4
(TRITON_MLA_SPARSE_TURBOQUANT, e5m2 on SM80), NO spec decode, 131k ctx.
Completion 21,993 tok in 432s = 50.9 t/s plain decode. finish=stop.
Score: 27/30 (judged by the serving session against docs/bwamem2-benchmark.md:
D1=2 stream-corruption identified but attributed toward hardware SDC rather
than the AVX-512 pipe-throughput class; D3=2 investigate-first emphasis;
D5=2 quickcheck hardening suggestion (rubric letter) though Nextflow retry
isolation itself stated correctly; D2/D4/D6/D7/D8/D9/D10=3).
bf16-KV baseline on the same tree (DFlash battery, same day): 28/30.
Delta is within judging noise -> no KV-quant quality regression signal.

---

## Overall read

Before the question-by-question review: both error signatures are **stream-corruption signatures**, not alignment-logic failures. Task 7 is most parsimoniously explained as the SAM stream ending mid-record (bwa-mem2 died or its output got mangled) — bwa-mem2 always emits a CIGAR field, even `*` for unmapped reads, so a well-formed-but-CIGAR-less record is unlikely. Task 8 is the header region mangled (the `SO:coordinate` failure is fallout from the same malformed header text, not a second failure). Two different corruption points in one run, intermittent, on a node that previously produced byte-identical work with the same inputs and containers, is a classic **silent-data-corruption pattern: RAM, NVMe, container image filesystem, or something that changed in the environment** — not a deterministic bwa-mem2 bug (a deterministic bug fails reproducibly and identically).

That changes the framing of your proposed fix: retry is a defensible stopgap, but the failure mode you can't retry your way out of is *silent* corruption — a bit-flip that produces a valid-looking BAM flows straight into MarkDuplicates and DeepVariant. A green retry on a suspect node is worse than a red X.

### 1. Retry vs. investigate

Both, but investigate is the primary response. The decisive, cheap experiment: **bwa-mem2 is deterministic for fixed input + thread count.** Re-run the exact command lines from `.command.sh` (failed task 7 or 8, plus a succeeded task) ~10× on H13-02 and once on H13-01, `md5sum` the output BAMs. Any cross-run divergence on H13-02, or H13-02 vs H13-01, is a smoking gun for hardware. You also already have two complete GIAB outputs from this exact data — use them as golden files for the same comparison.

Also, before trusting the error attribution: check whether `pipefail` is set in `.command.run`. Nextflow's default shell is `bash -ue` without pipefail; unless the module sets it, the exit code Nextflow saw was **samtools'**, and any bwa-mem2 crash (segfault, C++ exception, kill) is only visible in `.command.err`. You quoted samtools' stderr — go read bwa-mem2's.

### 2. Scope

Narrow — only `BWAMEM2_MEM`, as you have it. Broad retry policies convert deterministic bugs into flakiness and delay diagnosis. One caveat: with Nextflow running locally under Nomad placement, retries stay on the **same node**, so if this is hardware, retries are a coin flip on the same fault domain. Keep the `container` override, but pin by **digest** (`...@sha256:...`), not tag — a re-pushed tag at your local registry is one of the few software vectors that fits "same containers, different behavior."

### 3. One bug or two?

Almost certainly one mechanism, two symptoms. Within task 8, the header mangling causes the sort failure — that's one bug. Across tasks 7/8, random corruption hitting different offsets of the stream explains both. One suggestive detail: HLA alt contigs sit at the tail of the contig order in GRCh38-style references, so a truncated header or index write tends to surface there first — consistent, not proof.

This yields a unified alternative hypothesis worth testing: **corrupted staged inputs** (the `.ann`/`.bwt.2bit.64` index files or the FASTQ chunks for those two scatter tasks) in those tasks' work dirs. A truncated `.ann` produces exactly a missing-LN `@SQ` on a late contig. The failed work dirs still exist — checksum the staged index files and FASTQ chunks against the source and against a task-1–6 work dir *now*, before any cleanup runs. Corrupt staged file → storage/staging path. Pristine inputs → memory/compute path.

### 4. Retry-specific concerns

- **Stale outputs: not a risk.** Nextflow re-stages inputs per attempt and preserves the failed attempt's directory for debugging; partial outputs don't bleed into the retry.
- **The real risk is the opposite:** retry only catches corruption that fails *loudly*. On a node suspected of silent corruption, a successful retry tells you nothing about correctness.
- If input staging is the corruptor, the retry re-stages from the same source and can recur or corrupt differently.
- Cheap hardening regardless of root cause: add `samtools quickcheck -v` on the sorted BAM (module edit or `afterScript`). It catches the truncation class for nearly free; it won't catch subtle misalignment corruption, which is why the golden-file comparison matters.

### 5. Version pinning

2.2.1 is the current bwa-mem2 release, this is the standard bioconda build nf-core ships, and nothing in its issue history matches "intermittent stream corruption." Swapping versions won't fix a hardware fault and there's no newer version to swap to. What's actually worth verifying: image digests at the local registry (see #2) and the reference bundle checksums against the known sarek/iGenomes MD5s.

## Suggested diagnostic order

1. **Snapshot the failed task work dirs** (`.command.err`, `.command.out`, `.command.sh`, `.command.run`, `.exitcode`) before any Nomad/scratch cleanup touches them. Grep for bwa-mem2's own stderr; check for `pipefail` in `.command.run`.
2. **Checksum staged inputs** in the failed dirs vs. source vs. a successful task's dir (index files + FASTQ chunks).
3. **Determinism test** across nodes (or re-run the pipeline with `-resume` on H13-01 and H13-02 and compare alignment BAM md5s — keep thread counts identical).
4. **Hardware:** you checked dmesg for OOM, but also check MCE/EDAC/rasdaemon output, `nvme smart-log` (media errors, "percentage used"), and IPMI SEL. stressapptest/memtest in the next maintenance window.
5. **Environment diff since the last good run:** image digests, kernel/OS updates, and — importantly — the per-job work-directory change. "31 of 33 tasks ran fine in the new path" is weak exculpatory evidence for an intermittent fault. Is the new path on the *same device and filesystem* as the old one, same mount options? Is there any cleanup/GC process (Nomad or otherwise) that touches scratch mid-run? If the previous successful runs in your table predate the work-dir change, that change is a live suspect, not a ruled-out one.

Then decide: hardware → drain/repair H13-02, keep `retry` as the stopgap it is; software → fix and file upstream. The config snippet itself is fine as written — just don't let it be the whole answer.