# Prefix-cache quality probe: identical 32k prompt sent 4x (same nonce),
# report TTFT each time. Cold = full prefill; warm = cached-prefix turnaround.
# Also a "turn growth" variant: prompt + 3k extra tokens appended (agent turn).
import json, random, time, sys, urllib.request

BASE = "http://localhost:8080"
import os
KEY = os.environ["QWEN38FN_API_KEY"]  # export QWEN38FN_API_KEY first
H = {"Content-Type": "application/json", "Accept": "text/event-stream",
     "Authorization": "Bearer " + KEY}

def filler(approx_tokens, nonce):
    chars = int(approx_tokens * 1.763)
    rng = random.Random(nonce)
    out = []
    total = 0
    while total < chars:
        w = ("0x%08x sha%d:%040x cfg_%s=%d id%04X " % (
            rng.getrandbits(32), rng.randrange(2, 999), rng.getrandbits(160),
            "".join(rng.choice("abcdefghijklmnopqrstuvwxyz") for _ in range(6)),
            rng.getrandbits(12), rng.getrandbits(16)) +
            ".".join(str(rng.getrandbits(8)) for _ in range(4)) + " | ")
        out.append(w)
        total += len(w)
    return "".join(out)

TASK = ("Write a detailed step-by-step explanation of how a hash map works, "
        "including collision handling, resizing, and time complexity. Be thorough.")

def run(prompt, max_tokens=64):
    body = {"model": "qwen3.8", "messages": [{"role": "user", "content": prompt}],
            "max_tokens": max_tokens, "temperature": 0, "top_p": 1,
            "stream": True, "stream_options": {"include_usage": True},
            "chat_template_kwargs": {"enable_thinking": False}}
    req = urllib.request.Request(BASE + "/v1/chat/completions",
                                 data=json.dumps(body).encode(), headers=H)
    t0 = time.time(); t_first = None; ptok = 0; cached = 0
    with urllib.request.urlopen(req, timeout=1800) as r:
        for line in r:
            if not line.startswith(b"data: "):
                continue
            payload = line[6:].strip()
            if payload == b"[DONE]":
                break
            try:
                d = json.loads(payload)
            except Exception:
                continue
            if d.get("usage"):
                ptok = d["usage"].get("prompt_tokens") or ptok
                det = (d["usage"].get("prompt_tokens_details") or {})
                cached = det.get("cached_tokens") or 0
                continue
            ch = d.get("choices")
            if not ch:
                continue
            delta = ch[0].get("delta", {}) or {}
            if delta.get("content") and t_first is None:
                t_first = time.time()
    return ptok, cached, (t_first - t0 if t_first else -1)

if __name__ == "__main__":
    depth = int(sys.argv[1]) if len(sys.argv) > 1 else 32000
    base_prompt = filler(depth, f"cache{depth}") + "\n\nIgnore the records above. " + TASK
    for i in range(3):
        ptok, cached, ttft = run(base_prompt)
        print(f"  identical #{i}: prompt={ptok} cached={cached} TTFT={ttft:.2f}s")
    grown = filler(3000, f"growth{depth}") + "\nAssistant noted.\n" + base_prompt
    for i in range(2):
        ptok, cached, ttft = run(grown)
        print(f"  grown(+3k head) #{i}: prompt={ptok} cached={cached} TTFT={ttft:.2f}s")
    tail = base_prompt + filler(3000, f"tail{depth}")
    for i in range(2):
        ptok, cached, ttft = run(tail)
        print(f"  grown(+3k tail) #{i}: prompt={ptok} cached={cached} TTFT={ttft:.2f}s")
