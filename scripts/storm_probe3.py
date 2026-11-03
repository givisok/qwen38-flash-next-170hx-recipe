# Storm probe v3: TRUE-MISS neighbor — nonce FIRST (poisons the whole block
# hash chain), so every B send is a full ~25k prefill. This is the "neighbor
# agent cold-reads big new contexts" case. A = RU analysis, 1500 tokens.
import json, os, threading, time, urllib.request

BASE = "http://localhost:8080"
KEY = os.environ["QWEN38FN_API_KEY"]
H = {"Content-Type": "application/json", "Accept": "text/event-stream",
     "Authorization": "Bearer " + KEY}
RU = open("/root/qwen38fn/tools/ru_block.txt", encoding="utf-8").read()

A_TASK = ("Взгляни на релиз 1.4.2 (тикет LECTERAWEC-629, 57 сабтасок). Другой агент "
          "делает мерджи в main и не должен быть занят. Сделай развёрнутый анализ: "
          "как обычно выглядело распределение сабтасок по статусам перед финалом, "
          "какие типы задач обычно остаются немердженными и почему, как проверить "
          "полноту мерджей по пагинации, какие риски у релизной ветки. Подробно, "
          "по шагам. Только анализ, код не трогать.")

def body_true_miss(nonce):
    return (f"[ревизия {nonce}] " + (RU + "\n\n") * 14
            + "\n\nДай однострочный вывод: сколько примерно записей выше.")

def post(body, on_chunk=None):
    req = urllib.request.Request(BASE + "/v1/chat/completions", json.dumps(body).encode(), H)
    comp, tf, tl = 0, None, None
    with urllib.request.urlopen(req, timeout=900) as r:
        for raw in r:
            ln = raw.decode("utf-8", "replace").strip()
            if not ln.startswith("data: "): continue
            p = ln[6:]
            if p == "[DONE]": break
            try: j = json.loads(p)
            except Exception: continue
            if j.get("usage"):
                comp = j["usage"].get("completion_tokens") or comp
                continue
            ch = j.get("choices")
            if ch:
                d = ch[0].get("delta", {})
                if d.get("content"):
                    now = time.time()
                    if on_chunk: on_chunk(now)
                    tf, tl = tf or now, now
    return comp, tf, tl

def gen(task, maxtok, chunks=None):
    body = {"model": "qwen3.8", "messages": [{"role": "user", "content": task}],
            "max_tokens": maxtok, "temperature": 0.6, "top_p": 0.95, "stream": True,
            "stream_options": {"include_usage": True},
            "chat_template_kwargs": {"enable_thinking": False}}
    comp, tf, tl = post(body, on_chunk=(lambda t: chunks.append(t) if chunks is not None else None))
    return comp, (tl - tf) if tf and tl else 0

chunks, resA, wins = [], {}, []
def run_a():
    resA["c"], resA["d"] = gen(A_TASK, 1500, chunks)
ta = threading.Thread(target=run_a); t0 = time.time(); ta.start()
time.sleep(4)
for i in range(5):
    bs = time.time()
    comp, tf, tl = post({"model": "qwen3.8",
                         "messages": [{"role": "user", "content": body_true_miss(f"T{i}-{t0:.0f}")}],
                         "max_tokens": 16, "temperature": 0, "stream": True,
                         "stream_options": {"include_usage": True},
                         "chat_template_kwargs": {"enable_thinking": False}})
    be = time.time()
    wins.append((bs, be))
    print(f"  B send {i}: {be-bs:5.1f}s wall", flush=True)
    time.sleep(1.5)
ta.join()
inter = [t for t in chunks if any(a <= t <= b for a, b in wins)]
if len(inter) >= 2:
    dur = sum(min(b, inter[-1]) - max(a, inter[0]) for a, b in wins if b >= inter[0] and a <= inter[-1])
    in_tps = (len(inter) - 1) / dur if dur > 0 else 0
else:
    in_tps, dur = 0, 0
overall = (resA["c"] - 1) / resA["d"] if resA["d"] else 0
print(f"TRUE-MISS: A out {resA['c']} overall {overall:.1f} tok/s | during-B {in_tps:.1f} tok/s", flush=True)
# chunks outside B windows
outside = [t for t in chunks if not any(a <= t <= b for a, b in wins)]
if len(outside) >= 2:
    print(f"  (A between-B estimate: {(len([t for t in chunks]))} chunks total, {len(outside)} outside)", flush=True)
print("DONE", flush=True)
