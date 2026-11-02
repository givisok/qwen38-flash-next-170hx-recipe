# Spec-decode acceptance vs context depth: does the DFlash drafter die at the
# target's YaRN boundary (262144)? Probes two depths, samples spec counters
# around each request, reports accepted/draft and sparkDash decode tok/s.
import json, random, time, urllib.request, threading

BASE = "http://localhost:8080"
import os
KEY = os.environ["QWEN38FN_API_KEY"]  # export QWEN38FN_API_KEY first
H = {"Content-Type": "application/json", "Accept": "text/event-stream",
     "Authorization": "Bearer " + KEY}
MODEL = "qwen3.8"
CH_PER_TOK = 1.52025  # rec-filler, measured via /tokenize

def get(path):
    req = urllib.request.Request(BASE + path, headers=H)
    return urllib.request.urlopen(req, timeout=120).read()

def spec_counters():
    m = get("/metrics").decode()
    d, dt, at = 0, 0, 0
    for ln in m.splitlines():
        if ln.startswith("vllm:spec_decode_num_drafts_total"):
            d = float(ln.rsplit(" ", 1)[1])
        elif ln.startswith("vllm:spec_decode_num_draft_tokens_total"):
            dt = float(ln.rsplit(" ", 1)[1])
        elif ln.startswith("vllm:spec_decode_num_accepted_tokens_total"):
            at = float(ln.rsplit(" ", 1)[1])
    return d, dt, at

def tokenize_len(text):
    req = urllib.request.Request(BASE + "/tokenize",
                                 data=json.dumps({"model": MODEL, "prompt": text}).encode(), headers=H)
    return len(json.load(urllib.request.urlopen(req, timeout=600))["tokens"])

def build(target_tok, nonce):
    ratio = CH_PER_TOK
    for _ in range(3):
        rng = random.Random(nonce)
        rows, n, i = [], 0, 0
        while n < int(target_tok * ratio):
            w = ("rec %07d sku-%06x qty%03d bin %02d-%02d op%02x | " %
                 (i, rng.getrandbits(24), rng.getrandbits(10) % 500,
                  rng.getrandbits(7), rng.getrandbits(6), rng.getrandbits(8)))
            rows.append(w); n += len(w); i += 1
        doc = "".join(rows)
        t = tokenize_len(doc)
        if abs(t - target_tok) <= 4000:
            return doc, t
        ratio = ratio * target_tok / t
    raise SystemExit("calibration failed")

TASK = ("Write a detailed step-by-step explanation of how a B-tree index differs "
        "from an LSM-tree index, including write amplification and read paths. Keep writing.")

def probe(depth, nonce):
    doc, toks = build(depth, nonce)
    body = {"model": MODEL,
            "messages": [{"role": "user", "content": doc + "\n\nIgnore the records above. " + TASK}],
            "max_tokens": 200, "temperature": 1.0, "top_p": 0.95,
            "stream": True, "stream_options": {"include_usage": True},
            "chat_template_kwargs": {"enable_thinking": False}}
    req = urllib.request.Request(BASE + "/v1/chat/completions",
                                 data=json.dumps(body).encode(), headers=H)
    d0, dt0, at0 = spec_counters()
    t0 = time.time(); tfirst = None; tlast = None; comp = 0; ptok = 0
    with urllib.request.urlopen(req, timeout=1800) as r:
        for line in r:
            if not line.startswith(b"data: "):
                continue
            p = line[6:].strip()
            if p == b"[DONE]":
                break
            try:
                d = json.loads(p)
            except Exception:
                continue
            if d.get("usage"):
                comp = d["usage"].get("completion_tokens") or comp
                ptok = d["usage"].get("prompt_tokens") or ptok
                continue
            ch = d.get("choices")
            if ch and (ch[0].get("delta", {}) or {}).get("content"):
                now = time.time()
                tfirst = tfirst or now
                tlast = now
    d1, dt1, at1 = spec_counters()
    ddrafts, dacc = d1 - d0, at1 - at0
    dec = (comp - 1) / (tlast - tfirst) if comp and tlast and tlast > tfirst else 0
    acc_per_draft = dacc / ddrafts if ddrafts else 0
    print(f"@{ptok} tok: TTFT {tfirst-t0:.1f}s | decode {dec:.1f} tok/s | "
          f"drafts {int(ddrafts)} accepted {int(dacc)} ({dacc:.0f}) | "
          f"acc/draft {acc_per_draft:.2f}", flush=True)
    time.sleep(3)

for depth, nonce in ((250_000, "D250"), (270_000, "D270")):
    probe(depth, nonce)
print("healthy baseline reference: ~1.8 acc/draft, ~170 tok/s prose @32k", flush=True)
