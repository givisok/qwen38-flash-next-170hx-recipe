# Decode-at-depth, code output at real depths (calibrated filler 0.567 tok/char),
# plus prose. 2 rounds each. Usage-token based.
import json, time, sys, urllib.request

BASE = "http://localhost:8080"
KEY = "***REDACTED-API-KEY***"
H = {"Content-Type": "application/json", "Accept": "text/event-stream",
     "Authorization": "Bearer " + KEY}

def filler(approx_tokens, nonce):
    chars = int(approx_tokens * 1.763)
    rows = []
    i = 0
    body = (f"Inventory ledger {nonce}. SKU-{nonce}-%d widget, batch %d, "
            "qty %d, bin A-%d-%d, checked by operator %d. ")
    while sum(len(r) for r in rows) < chars:
        rows.append(body % (i, i % 97, (i * 7) % 50, i % 40, i % 20, i % 9))
        i += 1
    return "".join(rows)

CODE_TASK = ("binary_search_v9\n"
             "def binary_search(nums, target) -> int: index of target in a sorted list, or -1.\n"
             "Output only Python source. No comments, no docstrings, no markdown fences. "
             "Then add tests and the helpers this needs. Keep writing code.")
PROSE_TASK = ("Write a detailed step-by-step explanation of how a hash map works, "
              "including collision handling, resizing, and time complexity. Be thorough.")

def run_once(depth_tokens, task, nonce):
    prompt = filler(depth_tokens, nonce) + "\n\nIgnore the records above. " + task
    body = {"model": "qwen3.8", "messages": [{"role": "user", "content": prompt}],
            "max_tokens": 400, "temperature": 0, "top_p": 1,
            "stream": True, "stream_options": {"include_usage": True},
            "chat_template_kwargs": {"enable_thinking": False}}
    req = urllib.request.Request(BASE + "/v1/chat/completions",
                                 data=json.dumps(body).encode(), headers=H)
    t0 = time.time(); t_first = t_last = None; comp = 0; ptok = 0
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
                t_last = now
    ttft = t_first - t0 if t_first else 0
    tps = (comp - 1) / (t_last - t_first) if (comp and t_last and t_last > t_first) else 0
    return ptok, ttft, tps, comp

if __name__ == "__main__":
    depths = [int(x) for x in (sys.argv[1:] or ["32000", "64000", "128000"])]
    for task_name, task in (("code", CODE_TASK), ("prose", PROSE_TASK)):
        for dpt in depths:
            rates = []
            for rnd in range(2):
                ptok, ttft, tps, comp = run_once(dpt, task, f"{task_name}{dpt}R{rnd}{int(time.time())}")
                rates.append(tps)
                print(f"  {task_name} depth~{dpt}: prompt_tokens={ptok} TTFT={ttft:.2f}s "
                      f"decode={tps:.1f} tok/s (completion {comp})")
            print(f"{task_name} @{dpt}: median {sorted(rates)[len(rates)//2]:.1f} tok/s")
