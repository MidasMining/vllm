# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

# ruff: noqa: E501
"""Three-needle recall ladder against the GLM server (chat API, temp 0)."""

import json
import os
import random
import sys
import time
import urllib.request

from transformers import AutoTokenizer

LEVELS = [int(x) for x in sys.argv[1].split(",")]
OUT = sys.argv[2]

tok = AutoTokenizer.from_pretrained(
    os.environ.get("GLM_MODEL_DIR", "/mnt/ssd/models") + "/GLM-5.3-Flash-AWQ-W4A16",
    local_files_only=True,
)

rng = random.Random(11)
subjects = [
    "The survey team",
    "A maintenance crew",
    "The night operator",
    "An auditor",
    "The harbor office",
    "A field geologist",
    "The archive clerk",
    "The pump station",
]
verbs = [
    "recorded",
    "inspected",
    "catalogued",
    "reported",
    "measured",
    "logged",
    "reviewed",
]
objects = [
    "the east culvert",
    "a pallet of copper wire",
    "the tide gauge",
    "forty crates of grain",
    "the backup generator",
    "a cracked insulator",
    "the signal lamp",
    "the ledger for March",
]
tails = [
    "before noon.",
    "without incident.",
    "twice that week.",
    "as scheduled.",
    "after the storm passed.",
    "for the quarterly report.",
    "near the old bridge.",
]
sents = [
    f"{rng.choice(subjects)} {rng.choice(verbs)} {rng.choice(objects)} {rng.choice(tails)}"
    for _ in range(260000)
]
corpus_ids = tok.encode(" ".join(sents), add_special_tokens=False)
print("corpus tokens:", len(corpus_ids), flush=True)

NEEDLES = [
    (0.10, "IMPORTANT: the access code for the north gate is 48213.", "48213"),
    (0.50, "IMPORTANT: the lighthouse keeper's cat is named Pemberton.", "Pemberton"),
    (0.90, "IMPORTANT: the reactor was commissioned in the year 1987.", "1987"),
]
QUESTION = (
    "\n\nQuestion: Three IMPORTANT facts are hidden in the text above. What is the "
    "access code for the north gate, what is the lighthouse keeper's cat named, and "
    "in what year was the reactor commissioned? Answer in one short line."
)

for target in LEVELS:
    body = target - 150
    parts, prev = [], 0
    for frac, needle, _ in NEEDLES:
        cut = int(body * frac)
        parts.append(tok.decode(corpus_ids[prev:cut]))
        parts.append("\n" + needle + "\n")
        prev = cut
    parts.append(tok.decode(corpus_ids[prev:body]))
    prompt = "".join(parts) + QUESTION
    req = {
        "model": "glm53-flash",
        "messages": [{"role": "user", "content": prompt}],
        "temperature": 0,
        "max_tokens": 8192,
        "include_reasoning": False,
    }
    t0 = time.time()
    rec = {"target": target}
    try:
        d = json.load(
            urllib.request.urlopen(
                urllib.request.Request(
                    "http://localhost:8002/v1/chat/completions",
                    json.dumps(req).encode(),
                    {"Content-Type": "application/json"},
                ),
                timeout=7200,
            )
        )
        ans = d["choices"][0]["message"].get("content") or ""
        rec.update(
            prompt_tokens=d["usage"]["prompt_tokens"],
            completion_tokens=d["usage"]["completion_tokens"],
            finish=d["choices"][0].get("finish_reason"),
            answer=ans[:300],
            hits={k: (k in ans) for _, _, k in NEEDLES},
        )
    except Exception as e:
        rec["error"] = repr(e)[:300]
    rec["seconds"] = round(time.time() - t0, 1)
    print(json.dumps(rec), flush=True)
    with open(OUT, "a") as f:
        f.write(json.dumps(rec) + "\n")
    if "error" in rec:
        break
