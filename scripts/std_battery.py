# Standard gate battery for the qwen38fn lane: solo TG, N=6 aggregate,
# cold prefill (streamed TTFT), needle, greedy reproducibility. stdlib only.
import json
import statistics
import threading
import time
import urllib.request

BASE = "http://localhost:8080"
KEY = "***REDACTED-API-KEY***"
H = {"Content-Type": "application/json", "Authorization": "Bearer " + KEY}


def post(body, timeout=1800):
    req = urllib.request.Request(BASE + "/v1/completions", data=json.dumps(body).encode(), headers=H)
    with urllib.request.urlopen(req, timeout=timeout) as r:
        return json.loads(r.read().decode())


def post_stream(body, timeout=1800):
    """Return (ttft_s, total_s, n_tokens, text)."""
    req = urllib.request.Request(BASE + "/v1/completions", data=json.dumps(body).encode(), headers=H)
    t0 = time.time()
    ttft = None
    n = 0
    text = []
    with urllib.request.urlopen(req, timeout=timeout) as r:
        for raw in r:
            line = raw.decode().strip()
            if not line.startswith("data: ") or line == "data: [DONE]":
                continue
            d = json.loads(line[6:])
            if not d.get("choices"):
                n = n or (d.get("usage") or {}).get("completion_tokens") or 0
                continue
            ch = d["choices"][0]
            delta = ch.get("text") or ""
            if delta and ttft is None:
                ttft = time.time() - t0
            if delta:
                text.append(delta)
            if ch.get("finish_reason"):
                n = ch.get("completion_tokens") or d.get("usage", {}).get("completion_tokens") or 0
    total = time.time() - t0
    return ttft, total, n, "".join(text)


def metrics():
    txt = urllib.request.urlopen(BASE + "/metrics", timeout=60).read().decode()
    v = {}
    for line in txt.splitlines():
        for k in ("vllm:spec_decode_num_draft_tokens_total",
                  "vllm:spec_decode_num_accepted_tokens_total",
                  "vllm:spec_decode_num_drafts_total"):
            if line.startswith(k + "{"):
                v[k] = float(line.rsplit(" ", 1)[1])
    return v


def acc(m0, m1):
    dd = m1["vllm:spec_decode_num_draft_tokens_total"] - m0["vllm:spec_decode_num_draft_tokens_total"]
    da = m1["vllm:spec_decode_num_accepted_tokens_total"] - m0["vllm:spec_decode_num_accepted_tokens_total"]
    dn = m1["vllm:spec_decode_num_drafts_total"] - m0["vllm:spec_decode_num_drafts_total"]
    return dd, da, dn


def filler(target_tokens, nonce):
    para = ("Record {n}: the quarterly inventory of subsystem {m} lists {n} widgets, "
            "{n7} connectors, and {n3} fasteners, stored in aisle {m} bin {n7}. "
            "The auditor signed page {n3} on day {n} of month {m}. Ref {nonce}.")
    est = target_tokens * 4
    out, total, idx = [], 0, 0
    while total < est:
        n = idx % 977 + 3
        m = idx % 41 + 1
        n7 = (idx * 7) % 997 + 5
        n3 = (idx * 3) % 331 + 2
        s = para.format(n=n, m=m, n7=n7, n3=n3, nonce=nonce)
        out.append(s)
        total += len(s)
        idx += 1
    return " ".join(out)


TASKS = [
    ("py", "Write a complete Python implementation of an LRU cache class with get, put, and statistics methods, fully typed, with docstrings and a short demo. Output only code."),
    ("rust", "Write a Rust function that parses an ISO 8601 duration string (like P1DT2H30M) into seconds, with error handling and unit tests. Output only code."),
    ("edit", "Here is a Python function:\n\ndef fib(n):\n    if n < 2: return n\n    return fib(n-1) + fib(n-2)\n\nRewrite it as an iterative version with memoization removed, add type hints, input validation, and doctests. Output only code."),
    ("sql", "Write a PostgreSQL query with CTEs that finds the top 5 customers by revenue per region for 2025, handling refunds. Output only SQL."),
]

print("=== 1. SOLO TG (code battery, temp 0, 400 tok, x2 rounds) ===")
m00 = metrics()
res = {}
for rnd in range(2):
    for label, p in TASKS:
        ttft, total, n, _ = post_stream({"model": "qwen3.8", "prompt": p, "max_tokens": 400, "temperature": 0, "stream": True, "stream_options": {"include_usage": True}})
        res.setdefault(label, []).append((400 - 0 or n or 400) / (total - (ttft or 0)))
m01 = metrics()
dd, da, dn = acc(m00, m01)
for k, v in res.items():
    print(f"  {k}: {[round(x,1) for x in v]} median {round(statistics.median(v),1)} tok/s (decode-only)")
print(f"  battery acceptance: {100*da/dd:.1f}% | emitted/step {1+da/dn:.2f}")

print("=== 2. N=6 AGGREGATE (6 concurrent code tasks, 400 tok) ===")
m10 = metrics()
outs = [None] * 6
def worker(i):
    label, p = TASKS[i % len(TASKS)]
    _, total, _, _ = post_stream({"model": "qwen3.8", "prompt": p, "max_tokens": 400, "temperature": 0, "stream": True})
    outs[i] = total
th = [threading.Thread(target=worker, args=(i,)) for i in range(6)]
t0 = time.time()
for t in th: t.start()
for t in th: t.join()
wall = time.time() - t0
m11 = metrics()
dd, da, dn = acc(m10, m11)
print(f"  wall {wall:.1f}s | aggregate {6*400/wall:.0f} tok/s | acceptance {100*da/dd:.1f}% | emitted/step {1+da/dn:.2f}")

print("=== 3. COLD PREFILL 32k (x2, unique nonce each) ===")
for i in range(2):
    body = {"model": "qwen3.8", "prompt": filler(32000, f"PF{i}-{time.time()}") + "\n\nSummarize the inventory records above in one sentence.", "max_tokens": 40, "temperature": 0, "stream": True}
    ttft, total, n, _ = post_stream(body)
    print(f"  run{i}: TTFT {ttft:.2f}s | ~prefill {32000/max(ttft,0.01)/1000:.2f}k tok/s (approx tokens/TTFT)")

print("=== 4. NEEDLE at 64k ===")
secret = f"{int(time.time()) % 10000:04d}"
body = {"model": "qwen3.8", "prompt": filler(32000, "A") + f"\nThe secret access code for the north gate is {secret}. Remember it.\n" + filler(32000, "B") + "\n\nWhat is the secret access code for the north gate? Answer with the code only.", "max_tokens": 200, "temperature": 0}
r = post(body)
ans = r["choices"][0]["text"]
print(f"  secret={secret} | found={secret in ans} | answer head: {ans.strip()[:80]!r}")

print("=== 5. GREEDY x2 byte-identity ===")
outs2 = []
for i in range(2):
    _, _, _, txt = post_stream({"model": "qwen3.8", "prompt": TASKS[0][1], "max_tokens": 300, "temperature": 0, "stream": True})
    outs2.append(txt)
same = outs2[0] == outs2[1]
print(f"  byte-identical: {same} (len {len(outs2[0])} vs {len(outs2[1])})")
