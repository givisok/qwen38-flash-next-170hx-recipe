# pi-like probe with RANDOM (non-predictable) filler: temp 1.0, top_p 0.95,
# 32k+ context, long generation, windowed speed. Repetitive fillers inflate
# drafter acceptance; this one is near-unpredictable.
import json, random, time, sys, urllib.request

BASE = "http://localhost:8080"
KEY = "***REDACTED-API-KEY***"
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

CODE_TASK = ("binary_search_v9\n"
             "def binary_search(nums, target) -> int: index of target in a sorted list, or -1.\n"
             "Output only Python source. No comments, no docstrings, no markdown fences. "
             "Then add tests and the helpers this needs. Keep writing code.")
PROSE_TASK = ("Write a detailed step-by-step explanation of how a hash map works, "
              "including collision handling, resizing, and time complexity. Be thorough.")

def run_once(depth_tokens, task, nonce, temp, top_p):
    prompt = filler(depth_tokens, nonce) + "\n\nIgnore the records above. " + task
    body = {"model": "qwen3.8", "messages": [{"role": "user", "content": prompt}],
            "max_tokens": 900, "temperature": temp, "top_p": top_p,
            "stream": True, "stream_options": {"include_usage": True},
            "chat_template_kwargs": {"enable_thinking": False}}
    req = urllib.request.Request(BASE + "/v1/chat/completions",
                                 data=json.dumps(body).encode(), headers=H)
    t0 = time.time(); t_first = None; marks = []
    comp = 0; ptok = 0
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
                comp = d["usage"].get("completion_tokens") or comp
                ptok = d["usage"].get("prompt_tokens") or ptok
                continue
            ch = d.get("choices")
            if not ch:
                continue
            delta = ch[0].get("delta", {}) or {}
            if delta.get("content"):
                now = time.time()
                if t_first is None:
                    t_first = now
                marks.append(now)
    n = len(marks)
    if n > 2 and comp:
        out = []
        for lo, hi in ((0.0, 1/3), (1/3, 2/3), (2/3, 1.0)):
            a, b = int(n * lo), max(int(n * hi), int(n * lo) + 1)
            if b <= a or b > n:
                continue
            tok_a, tok_b = comp * a / n, comp * b / n
            dur = marks[min(b, n-1)] - marks[min(a, n-1)]
            if dur > 0:
                out.append((tok_b - tok_a) / dur)
        return ptok, t_first - t0, comp, out
    return ptok, (t_first - t0 if t_first else 0), comp, []

if __name__ == "__main__":
    temp, top_p = 1.0, 0.95
    depth = int(sys.argv[1]) if len(sys.argv) > 1 else 32000
    for task_name, task in (("code", CODE_TASK), ("prose", PROSE_TASK)):
        for rnd in range(2):
            ptok, ttft, comp, wins = run_once(depth, task, f"T1{task_name}{rnd}{int(time.time())}", temp, top_p)
            w = " | ".join(f"w{i+1}: {r:.1f}" for i, r in enumerate(wins))
            print(f"  {task_name} @{ptok}tok temp{temp}: TTFT {ttft:.2f}s, {comp} tok, windows tok/s: {w}")
