# Prefix-cache depth probe (2026-11-03): does OUR prod (MTP-4 on hybrid GDN)
# lose prefix-cache hits on replays, as upstream PR #52244 describes?
# Per size-tag: unique body, sent TWICE (identical) -> cached_tokens on replay
# + /metrics prefix-counter deltas. Thinking off, max_tokens 8, temp 0.
import json, os, re, time, urllib.request

BASE = "http://localhost:8080"
KEY = os.environ["QWEN38FN_API_KEY"]
H = {"Content-Type": "application/json", "Authorization": "Bearer " + KEY}
RU = open("/root/qwen38fn/tools/ru_block.txt", encoding="utf-8").read()

def send(content):
    body = {"model": "qwen3.8", "messages": [{"role": "user", "content": content}],
            "max_tokens": 8, "temperature": 0,
            "chat_template_kwargs": {"enable_thinking": False}}
    req = urllib.request.Request(BASE + "/v1/chat/completions",
                                 json.dumps(body).encode(), H)
    with urllib.request.urlopen(req, timeout=600) as r:
        j = json.load(r)
    u = j.get("usage", {})
    det = u.get("prompt_tokens_details") or {}
    return u.get("prompt_tokens"), det.get("cached_tokens")

def prefix_metrics():
    req = urllib.request.Request(BASE + "/metrics", None, H)
    d = {}
    with urllib.request.urlopen(req, timeout=30) as r:
        for ln in r.read().decode().splitlines():
            if ln.startswith("#") or "prefix" not in ln:
                continue
            m = re.match(r"^(\S+)\s+([\d.eE+-]+)$", ln.strip())
            if m:
                try:
                    d[m.group(1)] = float(m.group(2))
                except ValueError:
                    pass
    return d

send("ping")  # warmup canary
time.sleep(1)
for tag, chars in [("A", 12000), ("B", 13000), ("C", 18000), ("D", 26000)]:
    body = f"[проба кэша {tag} rev1] " + (RU * 8)[:chars]
    m0 = prefix_metrics()
    p1, c1 = send(body)
    time.sleep(1.5)
    m1 = prefix_metrics()
    p2, c2 = send(body)   # identical replay
    time.sleep(1.5)
    m2 = prefix_metrics()
    dm = {k: m2[k] - m1[k] for k in m2 if k in m1 and m2[k] != m1[k]}
    print(f"{tag}: ~{p1} tok | 1st cached {c1} | REPLAY cached {c2} "
          f"({100.0 * (c2 or 0) / max(p2 or 1, 1):.0f}% of prompt)", flush=True)
    for k, v in dm.items():
        print(f"    metrics delta replay: {k} +{v:.0f}", flush=True)
print("DONE", flush=True)
