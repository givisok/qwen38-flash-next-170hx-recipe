# Exact prefill measurement: uses usage.prompt_tokens, streams for TTFT.
import json
import time
import urllib.request

BASE = "http://localhost:8080"
KEY = "***REDACTED-API-KEY***"
H = {"Content-Type": "application/json", "Authorization": "Bearer " + KEY}


def filler(target_tokens, nonce):
    para = ("Record {n}: the quarterly inventory of subsystem {m} lists {n} widgets, "
            "{n7} connectors, and {n3} fasteners, stored in aisle {m} bin {n7}. "
            "The auditor signed page {n3} on day {n} of month {m}. Ref {nonce}.")
    est = target_tokens * 4
    out, total, idx = [], 0, 0
    while total < est:
        n = idx % 977 + 3; m = idx % 41 + 1
        n7 = (idx * 7) % 997 + 5; n3 = (idx * 3) % 331 + 2
        s = para.format(n=n, m=m, n7=n7, n3=n3, nonce=nonce)
        out.append(s); total += len(s); idx += 1
    return " ".join(out)


def probe(target, label):
    body = {"model": "qwen3.8",
            "prompt": filler(target, f"{label}-{time.time()}") + "\n\nSummarize the records above in one sentence.",
            "max_tokens": 30, "temperature": 0, "stream": True,
            "stream_options": {"include_usage": True}}
    req = urllib.request.Request(BASE + "/v1/completions", data=json.dumps(body).encode(), headers=H)
    t0 = time.time(); ttft = None; pt = None
    with urllib.request.urlopen(req, timeout=1800) as r:
        for raw in r:
            line = raw.decode().strip()
            if not line.startswith("data: ") or line == "data: [DONE]":
                continue
            d = json.loads(line[6:])
            if not d.get("choices"):
                pt = (d.get("usage") or {}).get("prompt_tokens")
                continue
            if d["choices"][0].get("text") and ttft is None:
                ttft = time.time() - t0
    print(f"{label}: prompt_tokens={pt} TTFT={ttft:.2f}s -> {pt/ttft/1000:.2f}k tok/s", flush=True)


for tgt, lab in [(8000, "p8k-a"), (8000, "p8k-b"), (32000, "p32k-a"), (32000, "p32k-b"), (65000, "p64k-a")]:
    probe(tgt, lab)
