#!/usr/bin/env python3
"""Concurrent decode test: N parallel streams, aggregate tok/s + acceptance window."""
import json, re, sys, time, urllib.request
from concurrent.futures import ThreadPoolExecutor

BASE = "http://<model-server-ip>:8080"
TAG = sys.argv[1] if len(sys.argv) > 1 else "conc"
N = int(sys.argv[2]) if len(sys.argv) > 2 else 6

def post(path, payload, timeout=900):
    req = urllib.request.Request(BASE + path, data=json.dumps(payload).encode(),
                                 headers={"Content-Type": "application/json"})
    t0 = time.time()
    d = json.loads(urllib.request.urlopen(req, timeout=timeout).read())
    return time.time() - t0, d

def metrics():
    agg = {}
    for line in urllib.request.urlopen(BASE + "/metrics", timeout=60).read().decode().splitlines():
        if not line.startswith("vllm:spec_decode_num_") or "created" in line:
            continue
        m = re.match(r'(vllm:\S+?)(\{[^}]*\})?\s+([0-9.eE+-]+)$', line)
        if not m:
            continue
        name, labels, val = m.group(1), m.group(2) or "", float(m.group(3))
        pos = ""
        pm = re.search(r'position="(\d+)"', labels)
        if pm:
            pos = pm.group(1)
        agg[name + pos] = agg.get(name + pos, 0.0) + val
    return agg

PROMPTS = [
    "Продолжай: высокий человек в сером плаще заметил у склада, где хранились старые карты. Он искал одну конкретную —",
    "Explain in detail how a mechanical watch escapement works, going gear by gear.",
    "Напиши подробный план подготовки трёхдневного похода в горы: снаряжение, еда, маршрут, безопасность.",
    "Write a Python function that finds the longest increasing subsequence, then explain its complexity step by step.",
    "Расскажи подробно об истории виноделия в Грузии от древности до наших дней.",
    "Describe the architecture of a modern CPU pipeline stage by stage, with details.",
    "Напиши эссе о том, почему океаны важны для климата планеты, с примерами течений.",
    "Explain how public-key cryptography works, step by step, with a simple analogy.",
]

def one(i):
    wall, d = post("/v1/chat/completions", {
        "model": "qwen3.8",
        "messages": [{"role": "user", "content": PROMPTS[i % len(PROMPTS)]}],
        "max_tokens": 256, "temperature": 0.0, "ignore_eos": True,
    })
    ct = d.get("usage", {}).get("completion_tokens")
    return wall, ct

out = {"tag": TAG, "n": N, "rounds": []}
for rnd in (1, 2):
    m0 = metrics()
    t0 = time.time()
    with ThreadPoolExecutor(N) as ex:
        res = list(ex.map(one, range(N)))
    wall_all = time.time() - t0
    m1 = metrics()
    tot_toks = sum(ct for _, ct in res)
    drafts = m1.get("vllm:spec_decode_num_drafts_total", 0) - m0.get("vllm:spec_decode_num_drafts_total", 0)
    atoks = m1.get("vllm:spec_decode_num_accepted_tokens_total", 0) - m0.get("vllm:spec_decode_num_accepted_tokens_total", 0)
    out["rounds"].append({
        "wall_max": round(max(w for w, _ in res), 2), "wall_all": round(wall_all, 2),
        "tokens": tot_toks, "aggregate_tok_s": round(tot_toks / wall_all, 1),
        "per_stream": [round(ct / w, 1) for w, ct in res],
        "tokens_per_step": round(1 + (atoks / drafts if drafts else 0), 3),
    })

fn = f"/tmp/lmhead_ab/{TAG}.json"
json.dump(out, open(fn, "w"), indent=1)
print(json.dumps(out["rounds"], ensure_ascii=False))
