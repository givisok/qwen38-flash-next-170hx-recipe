import json, random, time, urllib.request

BASE = "http://localhost:8080"
KEY = "***REDACTED-API-KEY***"
H = {"Content-Type": "application/json", "Authorization": "Bearer " + KEY}

def run(prompt):
    body = {"model": "qwen3.8", "messages": [{"role": "user", "content": prompt}],
            "max_tokens": 200, "temperature": 0, "top_p": 1,
            "stream": True, "stream_options": {"include_usage": True},
            "chat_template_kwargs": {"enable_thinking": False}}
    req = urllib.request.Request(BASE + "/v1/chat/completions",
                                 data=json.dumps(body).encode(), headers=H)
    out = []
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
            ch = d.get("choices")
            if ch and (ch[0].get("delta", {}) or {}).get("content"):
                out.append(ch[0]["delta"]["content"])
    return "".join(out)

rng = random.Random(120000)
secret = "%04d" % rng.randrange(10000)
chars = int(120000 * 1.763)
rows, i, total = [], 0, 0
while total < chars:
    w = ("rec %07d sku-%06x qty%03d bin %02d-%02d op%02x | " %
         (i, rng.getrandbits(24), rng.getrandbits(10) % 500,
          rng.getrandbits(7), rng.getrandbits(6), rng.getrandbits(8)))
    rows.append(w); total += len(w); i += 1
filler = "".join(rows)
# secret buried in the MIDDLE of a 120k-token wall of random records
half = len(filler) // 2
mid = filler[:half] + f"\nNOTE: the secret access code for the north gate is {secret}. \n" + filler[half:]
q = mid + "\n\nWhat was the secret access code for the north gate mentioned among the records? Answer with just the code."
t0 = time.time()
ans = run(q)
print(f"secret={secret} | found={secret in ans} | {time.time()-t0:.1f}s | head: {ans[:150]!r}")
