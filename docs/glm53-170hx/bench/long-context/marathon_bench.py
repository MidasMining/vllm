# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

# ruff: noqa: E501
"""Long-context battery for marathon use: speed at depth, cross-file tests at
depth, and an accumulated multi-turn soak. Chat API, temp 0, reasoning on.

Usage: marathon_bench.py OUT_JSONL [speed,xfile,soak]
"""

import json
import os
import random
import sys
import time
import urllib.request

import regex as re

sys.path.insert(0, os.environ.get("LLM_BENCH_DIR", "/home/user/llm-bench"))
from long_context_test import generate_code_files  # noqa: E402
from transformers import AutoTokenizer  # noqa: E402

OUT = sys.argv[1]
LEGS = (sys.argv[2] if len(sys.argv) > 2 else "speed,xfile,soak").split(",")
URL = "http://localhost:8002/v1/chat/completions"
DEPTHS = [131072, 524288, 1040000]

tok = AutoTokenizer.from_pretrained(
    os.environ.get("GLM_MODEL_DIR", "/mnt/ssd/models") + "/GLM-5.3-Flash-AWQ-W4A16",
    local_files_only=True,
)
ntok = lambda s: len(tok.encode(s, add_special_tokens=False))  # noqa: E731


def log(rec):
    print(json.dumps(rec), flush=True)
    with open(OUT, "a") as f:
        f.write(json.dumps(rec) + "\n")


# Realistic filler: ops/engineering log prose with varied numbers, so it is
# not trivially compressible and contains distractor numerals.
rng = random.Random(23)
systems = [
    "the stratum proxy",
    "the payout daemon",
    "the share validator",
    "the redis cache",
    "the block notifier",
    "the PPLNS window",
    "the web dashboard",
    "the node RPC",
    "the difficulty retargeter",
    "the backup cron job",
    "the metrics exporter",
]
events = [
    "restarted after a config reload",
    "logged a transient timeout",
    "reported elevated latency",
    "completed a scheduled compaction",
    "rotated its log files",
    "rejected a malformed request",
    "reconnected to the upstream node",
    "flushed a batch of pending writes",
    "passed its health check",
    "emitted a deprecation warning",
]
paras = []
for i in range(120000):
    paras.append(
        f"[{rng.randint(0, 23):02d}:{rng.randint(0, 59):02d}] {rng.choice(systems)} "
        f"{rng.choice(events)} (worker {rng.randint(1, 512)}, "
        f"{rng.randint(10, 9999)} ms, queue depth {rng.randint(0, 300)})."
    )
FILLER_IDS = tok.encode("\n".join(paras), add_special_tokens=False)
print("filler tokens:", len(FILLER_IDS), flush=True)
_fill_off = [0]


def filler(n):
    """Next n filler tokens (advancing, wrapping) decoded to text."""
    s = _fill_off[0] % (len(FILLER_IDS) - n - 1)
    _fill_off[0] = s + n
    return tok.decode(FILLER_IDS[s : s + n])


def chat(messages, max_tokens, stream=True, timeout=7200):
    """Streaming chat call. Returns timing + text; counts reasoning deltas as
    output so TTFT and decode rate reflect what the GPU is doing."""
    req = {
        "model": "glm53-flash",
        "messages": messages,
        "temperature": 0,
        "max_tokens": max_tokens,
        "stream": True,
        "stream_options": {"include_usage": True},
    }
    t0 = time.time()
    t_first = None
    content, reasoning, usage, finish = [], [], None, None
    r = urllib.request.urlopen(
        urllib.request.Request(
            URL, json.dumps(req).encode(), {"Content-Type": "application/json"}
        ),
        timeout=timeout,
    )
    for raw in r:
        line = raw.decode().strip()
        if not line.startswith("data: ") or line == "data: [DONE]":
            continue
        d = json.loads(line[6:])
        if d.get("usage"):
            usage = d["usage"]
        for ch in d.get("choices", []):
            delta = ch.get("delta", {})
            c = delta.get("content") or ""
            rz = delta.get("reasoning_content") or delta.get("reasoning") or ""
            if (c or rz) and t_first is None:
                t_first = time.time()
            content.append(c)
            reasoning.append(rz)
            if ch.get("finish_reason"):
                finish = ch["finish_reason"]
    t_end = time.time()
    comp = usage["completion_tokens"] if usage else None
    dec = None
    if comp and t_first and comp > 1 and t_end > t_first:
        dec = round((comp - 1) / (t_end - t_first), 1)
    return {
        "prompt_tokens": usage["prompt_tokens"] if usage else None,
        "completion_tokens": comp,
        "ttft_s": round((t_first or t_end) - t0, 1),
        "total_s": round(t_end - t0, 1),
        "decode_tps": dec,
        "finish": finish,
        "content": "".join(content),
        "reasoning": "".join(reasoning),
    }


def loopiness(text, n=8):
    """Fraction of repeated word 8-grams — high values flag degenerate loops."""
    w = text.split()
    grams = [" ".join(w[i : i + n]) for i in range(max(0, len(w) - n))]
    return round(1 - len(set(grams)) / len(grams), 3) if grams else 0.0


# ---------------------------------------------------------------- 1. speed
def leg_speed():
    for depth in DEPTHS:
        body = filler(depth - 200)
        q1 = (
            "\n\nThe text above is an operations log. Ignore it for now and write an "
            "original ~900-word short story about a lighthouse keeper."
        )
        m = [{"role": "user", "content": body + q1}]
        r1 = chat(m, 4096)
        log(
            {
                "leg": "speed",
                "depth": depth,
                "turn": 1,
                **{
                    k: r1[k]
                    for k in (
                        "prompt_tokens",
                        "completion_tokens",
                        "ttft_s",
                        "total_s",
                        "decode_tps",
                        "finish",
                    )
                },
            }
        )
        m += [
            {"role": "assistant", "content": r1["content"]},
            {"role": "user", "content": "Continue the story for another ~500 words."},
        ]
        r2 = chat(m, 4096)
        log(
            {
                "leg": "speed",
                "depth": depth,
                "turn": 2,
                **{
                    k: r2[k]
                    for k in (
                        "prompt_tokens",
                        "completion_tokens",
                        "ttft_s",
                        "total_s",
                        "decode_tps",
                        "finish",
                    )
                },
            }
        )


# ---------------------------------------------------------------- 2. xfile
FILES = generate_code_files()


def scattered(depth):
    """Code files spread across 10..90% depth inside filler."""
    names = list(FILES)
    parts_tok = sum(ntok(f"=== {n} ===\n{FILES[n]}") for n in names)
    budget = depth - parts_tok - 400
    fracs = [0.1 + 0.8 * i / max(1, len(names) - 1) for i in range(len(names))]
    out, prev = [], 0
    for n, f in zip(names, fracs):
        cut = int(budget * f)
        out.append(filler(cut - prev))
        out.append(f"\n\n=== {n} ===\n{FILES[n]}\n\n")
        prev = cut
    out.append(filler(budget - prev))
    return "".join(out)


def leg_xfile():
    for depth in DEPTHS:
        ctx = scattered(depth)
        q = (
            f"Below is an operations log with the source files of a cryptocurrency mining "
            f"pool embedded in it (each starts with '=== filename ===').\n\n{ctx}\n\n"
            "Question: Trace the flow when a miner submits a share that finds a valid block. "
            "What happens step by step from share submission to payout queuing? Be specific "
            "about which methods are called and in which files. Answer concisely."
        )
        r = chat([{"role": "user", "content": q}], 8192)
        keys = [
            "handle_submit",
            "validate_share",
            "submit_block",
            "process_block",
            "pplns",
            "payout",
        ]
        found = [k for k in keys if k in r["content"].lower()]
        log(
            {
                "leg": "xfile",
                "test": "multi_file",
                "depth": depth,
                "prompt_tokens": r["prompt_tokens"],
                "ttft_s": r["ttft_s"],
                "score": f"{len(found)}/{len(keys)}",
                "found": found,
                "finish": r["finish"],
                "answer": r["content"][:600],
            }
        )

        ctx = scattered(depth)
        q = (
            f"Below is an operations log with mining pool source files embedded in it.\n\n"
            f"{ctx}\n\nAnswer from the source code:\n"
            "1. What is the PPLNS N multiplier default value?\n"
            "2. What is the minimum payout threshold?\n"
            "3. Is block hash byte order reversed for comparison? (yes/no)\n"
            "4. What port does the stratum server listen on by default?\n"
            "Format exactly as four lines:\n1. [value]\n2. [value]\n3. [yes/no]\n4. [port]"
        )
        r = chat([{"role": "user", "content": q}], 8192)
        lines = {
            m.group(1): m.group(2).strip().lower()
            for m in re.finditer(r"^\s*([1-4])\.\s*(.+)$", r["content"], re.M)
        }
        checks = {
            "n_multiplier": bool(re.search(r"\b2(\.0)?\b", lines.get("1", ""))),
            "min_payout": bool(re.search(r"\b1(\.0)?\b", lines.get("2", ""))),
            "byte_order": lines.get("3", "").startswith("yes"),
            "port": "3333" in lines.get("4", ""),
        }
        log(
            {
                "leg": "xfile",
                "test": "cross_ref",
                "depth": depth,
                "prompt_tokens": r["prompt_tokens"],
                "ttft_s": r["ttft_s"],
                "score": f"{sum(checks.values())}/4",
                "checks": checks,
                "finish": r["finish"],
                "answer": r["content"][:300],
            }
        )


# ---------------------------------------------------------------- 3. soak
def leg_soak(turns=11, chunk=45000):
    """Accumulated session: each turn appends a ~45k-token chapter with one
    fresh fact, asks for it plus a fact from chapter 1. Assistant replies are
    fed back as content only (reasoning dropped, as standard clients do)."""
    facts = [(f"batch {i}", str(rng.randint(10000, 99999))) for i in range(turns)]
    msgs = []
    for t in range(turns):
        name, val = facts[t]
        half = chunk // 2
        chapter = (
            f"--- CHAPTER {t + 1} ---\n"
            + filler(half)
            + f"\nRECORD: the calibration constant for {name} is {val}.\n"
            + filler(chunk - half)
        )
        ask = (
            f"\n\nQuestion: what is the calibration constant for {name}, and what is the "
            f"calibration constant for {facts[0][0]}? Reply in one short line."
        )
        msgs.append({"role": "user", "content": chapter + ask})
        r = chat(msgs, 8192)
        ans = r["content"]
        log(
            {
                "leg": "soak",
                "turn": t + 1,
                "prompt_tokens": r["prompt_tokens"],
                "ttft_s": r["ttft_s"],
                "decode_tps": r["decode_tps"],
                "completion_tokens": r["completion_tokens"],
                "finish": r["finish"],
                "fresh_ok": val in ans,
                "first_ok": facts[0][1] in ans,
                "loop_content": loopiness(ans),
                "loop_reasoning": loopiness(r["reasoning"]),
                "answer": ans[:200],
            }
        )
        msgs.append({"role": "assistant", "content": ans})


for leg in LEGS:
    {"speed": leg_speed, "xfile": leg_xfile, "soak": leg_soak}[leg]()
print("MARATHON-DONE", flush=True)
