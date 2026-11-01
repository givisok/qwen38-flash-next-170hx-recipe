# Full decode-speed sweep across fill depth: 32k/64k/128k/256k/512k/1M.
# Same method as spec_gate_depth.py: self-calibrated rec-filler, temp 1,
# 200-token decode, spec-counter deltas around each request, plus a
# quiet-check (waits for 0 running requests) so foreign traffic does not
# pollute the global draft/accept counters.
import json, random, time, urllib.request

BASE = "http://localhost:8080"
KEY  = "***REDACTED-API-KEY***"
H = {"Content-Type": "application/json", "Accept": "text/event-stream",
     "Authorization": "Bearer " + KEY}
MODEL = "qwen3.8"
CH_PER_TOK = 1.52025  # rec-filler, measured via /tokenize
DEPTHS = [32_000, 64_000, 128_000, 256_000, 512_000, 1_000_000]
YAARN_AT = 262_144

def metrics():
    m = urllib.request.urlopen(urllib.request.Request(BASE + "/metrics", headers=H),
                               timeout=120).read().decode()
    d = {"drafts": 0.0, "draft_tokens": 0.0, "accepted": 0.0, "running": 0.0}
    for ln in m.splitlines():
        if ln.startswith("vllm:spec_decode_num_drafts_total"):
            d["drafts"] = float(ln.rsplit(" ", 1)[1])
        elif ln.startswith("vllm:spec_decode_num_draft_tokens_total"):
            d["draft_tokens"] = float(ln.rsplit(" ", 1)[1])
        elif ln.startswith("vllm:spec_decode_num_accepted_tokens_total"):
            d["accepted"] = float(ln.rsplit(" ", 1)[1])
        elif ln.startswith("vllm:num_requests_running{"):
            d["running"] = float(ln.rsplit(" ", 1)[1])
    return d

def wait_quiet(timeout=180):
    t0 = time.time()
    while time.time() - t0 < timeout:
        if metrics()["running"] == 0:
            return True
        time.sleep(5)
    return False

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
        if abs(t - target_tok) <= max(4000, target_tok * 0.005):
            return doc, t
        ratio = ratio * target_tok / t
    raise SystemExit(f"calibration failed at {target_tok}")

TASK = ("Write a detailed step-by-step explanation of how a B-tree index differs "
        "from an LSM-tree index, including write amplification and read paths. Keep writing.")

def probe(depth, nonce):
    doc, toks = build(depth, nonce)
    if not wait_quiet():
        print(f"@{depth}: SKIPPED - server busy for 180s", flush=True)
        return
    body = {"model": MODEL,
            "messages": [{"role": "user", "content": doc + "\n\nIgnore the records above. " + TASK}],
            "max_tokens": 200, "temperature": 1.0, "top_p": 0.95,
            "stream": True, "stream_options": {"include_usage": True},
            "chat_template_kwargs": {"enable_thinking": False}}
    req = urllib.request.Request(BASE + "/v1/chat/completions",
                                 data=json.dumps(body).encode(), headers=H)
    m0 = metrics()
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
    m1 = metrics()
    dd, da = m1["drafts"] - m0["drafts"], m1["accepted"] - m0["accepted"]
    dec = (comp - 1) / (tlast - tfirst) if comp and tlast and tlast > tfirst else 0
    ttft = tfirst - t0 if tfirst else 0
    pre = ptok / ttft if ttft else 0
    side = "below" if ptok < YAARN_AT else "ABOVE"
    print(f"@{ptok:>8} tok ({side} yarn): TTFT {ttft:6.1f}s prefill {pre:6.0f} tok/s | "
          f"decode {dec:6.1f} tok/s | drafts {int(dd):4d} acc {int(da):4d} | "
          f"acc/draft {da/dd if dd else 0:4.2f} | running b{int(m0['running'])}/a{int(m1['running'])}",
          flush=True)
    time.sleep(3)

print("depth sweep (prose, temp 1, 200 tok):", flush=True)
for depth in DEPTHS:
    probe(depth, f"SW{depth}")
