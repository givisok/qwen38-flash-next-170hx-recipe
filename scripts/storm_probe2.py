# Storm probe v2 (2026-11-03, prod MTP-4 + prefix-replay-depth fix):
# phases SOLO / STORM-MISS (neighbor prefill, unique prefix each send) /
# STORM-HIT (neighbor resends the SAME prompt -> 100%-depth replays after the
# fix) / PAIR (two decoders, no prefill). A = RU analysis stream, chunks
# timestamped for in-window tps.
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

def body_miss(nonce):
    body = (RU + "\n\n") * 14
    return body + f"\n\nКонтекст-ревизия {nonce}. Дай однострочный вывод: сколько примерно записей выше. Nonce {nonce}."

BODY_HIT = (RU + "\n\n") * 14 + "\n\nДай однострочный вывод: сколько примерно записей выше."

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

def breq(content):
    return {"model": "qwen3.8", "messages": [{"role": "user", "content": content}],
            "max_tokens": 16, "temperature": 0, "stream": True,
            "stream_options": {"include_usage": True},
            "chat_template_kwargs": {"enable_thinking": False}}

def gen(task, maxtok, chunks=None):
    body = {"model": "qwen3.8", "messages": [{"role": "user", "content": task}],
            "max_tokens": maxtok, "temperature": 0.6, "top_p": 0.95, "stream": True,
            "stream_options": {"include_usage": True},
            "chat_template_kwargs": {"enable_thinking": False}}
    comp, tf, tl = post(body, on_chunk=(lambda t: chunks.append(t) if chunks is not None else None))
    return comp, (tl - tf) if tf and tl else 0

def tps_in(chunks, windows):
    inter = [t for t in chunks if any(a <= t <= b for a, b in windows)]
    if len(inter) < 2: return 0.0, 0
    dur = sum(min(b, inter[-1]) - max(a, inter[0]) for a, b in windows if b >= inter[0] and a <= inter[-1])
    return (len(inter) - 1) / dur if dur > 0 else 0, len(inter)

def run_storm(tag, make_prompt):
    chunks, resA, wins = [], {}, []
    def run_a():
        resA["c"], resA["d"] = gen(A_TASK, 1500, chunks)
    ta = threading.Thread(target=run_a); t0 = time.time(); ta.start()
    time.sleep(4)
    for i in range(5):
        bs = time.time()
        comp, tf, tl = post(breq(make_prompt(i, t0)))
        be = time.time()
        wins.append((bs, be))
        print(f"  B send {i}: {be-bs:5.1f}s wall", flush=True)
        time.sleep(1.5)
    ta.join()
    in_tps, n_in = tps_in(chunks, wins)
    overall = (resA["c"] - 1) / resA["d"] if resA["d"] else 0
    print(f"{tag}: A out {resA['c']} overall {overall:.1f} tok/s | during-B {in_tps:.1f} tok/s (n={n_in})", flush=True)
    time.sleep(3)

# ---- phase 1: SOLO
c, d = gen(A_TASK, 500)
print(f"SOLO        : out {c} | {(c-1)/d if d else 0:.1f} tok/s", flush=True)
time.sleep(3)

# ---- phase 2: STORM-MISS (unique prefix per send)
run_storm("STORM-MISS ", lambda i, t0: body_miss(f"M{i}-{t0:.0f}"))

# ---- phase 3: STORM-HIT (identical resends -> replays)
run_storm("STORM-HIT  ", lambda i, t0: BODY_HIT)

# ---- phase 4: PAIR (two decoders, no prefill)
res = {}
def worker(tag):
    res[tag] = gen(A_TASK + f" Вариант {tag}.", 700)
ths = [threading.Thread(target=worker, args=(t,)) for t in ("X", "Y")]
[t.start() for t in ths]; [t.join() for t in ths]
for tag, (c, d) in res.items():
    print(f"PAIR {tag}      : out {c} | {(c-1)/d if d else 0:.1f} tok/s", flush=True)
print("DONE", flush=True)
