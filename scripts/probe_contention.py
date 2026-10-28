#!/usr/bin/env python3
"""Contention probe: streaming decode A + late 120k prefill B.
Measures A's inter-token gaps during B's prefill window and B's TTFT."""
import json, re, sys, time, urllib.request
from threading import Thread

BASE = "http://<model-server-ip>:8080"
TAG = sys.argv[1] if len(sys.argv) > 1 else "probe"
B_DELAY = 2.0

def post_stream(payload, sink, timeout=600):
    """POST with stream=True; call sink(text_chunk) per delta; return (t_first, t_last, n_chunks)."""
    req = urllib.request.Request(BASE + "/v1/chat/completions",
                                 data=json.dumps(payload).encode(),
                                 headers={"Content-Type": "application/json"})
    t0 = time.time()
    resp = urllib.request.urlopen(req, timeout=timeout)
    t_first = None
    n = 0
    for raw in resp:
        line = raw.decode("utf-8", "replace").strip()
        if not line.startswith("data: "):
            continue
        data = line[6:]
        if data == "[DONE]":
            break
        d = json.loads(data)
        delta = d["choices"][0].get("delta", {})
        txt = delta.get("content") or ""
        now = time.time()
        if t_first is None:
            t_first = now
        sink(txt, now)
        n += 1
    return t0, t_first, n

res = {}

def run_a():
    events = []
    t0, t_first, n = post_stream({
        "model": "qwen3.8",
        "messages": [{"role": "user", "content": "Напиши максимально подробную историю мореплавания от античности до XIX века: финикийцы, греки, викинги, эпоха великих географических открытий, кругосветки. Длинно, с датами и именами."}],
        "max_tokens": 900, "temperature": 0.0, "ignore_eos": True, "stream": True,
    }, lambda txt, ts: events.append((ts, txt)))
    res["A"] = {"events": events, "t0": t0, "t_first": t_first, "n_chunks": n}

def run_b():
    nonce = f"Партия {int(time.time()*1000)}. "
    big = nonce + ("В 1519 году флотилия Магеллана вышла из Санлукара. Моряки не знали, что поход продлится три года. "
                   "Запасы были скудны, карты неточны, но курс лежал на запад. ") * 3400
    t0, t_first, n = post_stream({
        "model": "qwen3.8",
        "messages": [{"role": "user", "content": big + "\n\nВопрос: в каком году флотилия вышла из Санлукара? Одной строкой."}],
        "max_tokens": 48, "temperature": 0.0, "stream": True,
    }, lambda txt, ts: None)
    res["B"] = {"t0": t0, "t_first": t_first,
                "ttft": round(t_first - t0, 2) if t_first else None}

ta = Thread(target=run_a); ta.start()
time.sleep(B_DELAY)
tb = Thread(target=run_b); tb.start()
ta.join(); tb.join()

ev = res["A"]["events"]
toks = [t for t, _ in ev]
b0, b1 = res["B"]["t0"], res["B"]["t_first"]
window = [(t, txt) for t, txt in ev if b0 <= t <= (b1 or ev[-1][0])]
wt = [t for t, _ in window]
gaps_all = [round(b - a, 3) for a, b in zip(toks, toks[1:])]
gaps_win = [round(b - a, 3) for a, b in zip(wt, wt[1:])] if len(wt) > 1 else []

out = {
    "tag": TAG, "ts": time.strftime("%F %T"),
    "B_ttft_s": res["B"]["ttft"],
    "A_chunks": len(ev),
    "A_window_chunks": len(window),
    "A_gaps_window": {
        "max": max(gaps_win) if gaps_win else None,
        "over_0p5s": sum(1 for g in gaps_win if g > 0.5),
        "over_1s": sum(1 for g in gaps_win if g > 1.0),
        "over_1p5s": sum(1 for g in gaps_win if g > 1.5),
        "n": len(gaps_win),
    },
    "A_gaps_all_max": max(gaps_all) if gaps_all else None,
    "A_effective_tok_s": round(len(ev) / (toks[-1] - toks[0]), 1) if len(toks) > 2 else None,
}
fn = f"/tmp/lmhead_ab/{TAG}.json"
json.dump(out, open(fn, "w"), indent=1)
print(json.dumps(out))
