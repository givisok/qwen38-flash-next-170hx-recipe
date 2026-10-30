#!/usr/bin/env python3
"""Incremental-prefill TTFT battery for the PLE prefill-checkpoint port.

Each cell primes a nonce'd prompt of P tokens, then sends the same prompt grown
by G tokens: that request's prefill crosses the last Mamba-checkpoint boundary.
With internal prefill checkpoints OFF (baseline) the final chunk stops at the
hash-aligned checkpoint position and the tail re-runs as an extra pipeline step;
with them ON the prefill completes in one step. max_tokens=1 keeps the wall time
prefill-dominated. Solo requests only (no FN12 interference).
"""
import json, random, sys, time, urllib.request

BASE = "http://192.168.1.12:8080"
TAG = sys.argv[1] if len(sys.argv) > 1 else "run"

FILLER = ("The quick brown fox jumps over the lazy dog near the river bank. "
          "Photovoltaic efficiency measurements were recorded quarterly. ") * 400  # ~5.5k tok

def chat(content, max_tokens=1):
    req = urllib.request.Request(BASE + "/v1/chat/completions", method="POST",
        headers={"Content-Type": "application/json"},
        data=json.dumps({"model": "qwen3.8", "messages": [{"role": "user", "content": content}],
                         "max_tokens": max_tokens, "temperature": 0,
                         "ignore_eos": True}).encode())
    t0 = time.time()
    with urllib.request.urlopen(req, timeout=600) as r:
        d = json.loads(r.read())
    dt = time.time() - t0
    u = d["usage"]
    cached = (u.get("prompt_tokens_details") or {}).get("cached_tokens", 0)
    return {"wall": round(dt, 3), "prompt_tokens": u.get("prompt_tokens"),
            "cached": cached, "completion_tokens": u.get("completion_tokens")}

def approx(prompt_tokens, text):
    # scale filler to requested token count (~1.35 tok/word heuristics is unstable;
    # we simply report the actual usage the server measured)
    return text

out = {"tag": TAG, "ts": time.strftime("%Y-%m-%d %H:%M:%S"), "cells": {}, "cold": []}

# --- prime+grow cells: P chosen on 1728-boundaries, G crosses the next boundary
for (P, G) in [(8640, 900), (8640, 1729), (17280, 3000), (8640, 300)]:
    nonce = random.randint(10**8, 10**9)
    base_txt = f"Session {nonce}. " + FILLER * max(1, P // 5500)
    prime = chat(base_txt)
    grow = chat(base_txt + f"\nAdditional note {nonce}: please summarize nothing, just acknowledge. " + FILLER[: G * 6])
    new_tok = grow["prompt_tokens"] - grow["cached"]
    cell = {"P_target": P, "G_target": G, "prime": prime, "grow": grow,
            "incremental_new_tokens": new_tok,
            "incremental_tok_s": round(new_tok / grow["wall"], 1) if grow["wall"] else None}
    out["cells"][f"P{P}_G{G}"] = cell
    print(f"P{P}+G{G}: prime {prime['wall']}s (cached {prime['cached']}), "
          f"grow {grow['wall']}s, new={new_tok}, {cell['incremental_tok_s']} tok/s")

# --- cold solo single-chunk prefill (8000 < 8192 budget)
for i in range(3):
    nonce = random.randint(10**8, 10**9)
    txt = f"Cold {nonce}. " + FILLER * 2 + FILLER[: 12000]
    r = chat(txt)
    out["cold"].append(r)
    print(f"cold{i}: {r['wall']}s for {r['prompt_tokens']} tok = "
          f"{round(r['prompt_tokens']/r['wall'],1)} tok/s (cached {r['cached']})")

fn = f"/tmp/lmhead_ab/ple_ckpt_{TAG}.json"
json.dump(out, open(fn, "w"), indent=1)
print("saved", fn)
