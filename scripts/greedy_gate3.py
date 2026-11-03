# Greedy identity x3 + a mid-page resume correctness check:
# 1) same code prompt twice (2nd replays from the deep landing) x3 pairs;
# 2) cold vs deep-replay RU answer sanity (needle-style exact secret recall).
import json, os, time, urllib.request

BASE = "http://localhost:8080"
KEY = os.environ["QWEN38FN_API_KEY"]
H = {"Content-Type": "application/json", "Authorization": "Bearer " + KEY}
RU = open("/root/qwen38fn/tools/ru_block.txt", encoding="utf-8").read()

CODE = ("Напиши на python функцию binary_search(arr, x) с докстрингом и "
        "тремя примерами использования. Только код, без пояснений.")

def comp(prompt, maxtok=300):
    body = {"model": "qwen3.8", "prompt": prompt, "max_tokens": maxtok,
            "temperature": 0, "stream": True}
    req = urllib.request.Request(BASE + "/v1/completions",
                                 json.dumps(body).encode(), H)
    txt = []
    with urllib.request.urlopen(req, timeout=600) as r:
        for raw in r:
            ln = raw.decode("utf-8", "replace").strip()
            if not ln.startswith("data: "):
                continue
            p = ln[6:]
            if p == "[DONE]":
                break
            try:
                j = json.loads(p)
            except Exception:
                continue
            ch = j.get("choices")
            if ch and ch[0].get("text"):
                txt.append(ch[0]["text"])
    return "".join(txt)

ok = 0
for i in range(3):
    a = comp(CODE)
    time.sleep(1)
    b = comp(CODE)
    same = a == b
    ok += same
    print(f"pair {i}: identical={same} (len {len(a)} vs {len(b)})", flush=True)
print(f"greedy identity: {ok}/3", flush=True)

# deep-replay exact recall: secret buried at ~7k tokens, ask after replay
secret = f"CODE-{int(time.time()) % 100000:05d}"
big = ("Фоновый текст. " * 60 + f"\n\nСекретный код доступа к северным воротам: {secret}.\n\n"
       + (RU * 3)[:14000] + f"\n\nВопрос: назови точный секретный код доступа к северным воротам. Код: ")
def ask(content):
    body = {"model": "qwen3.8", "messages": [{"role": "user", "content": content}],
            "max_tokens": 60, "temperature": 0,
            "chat_template_kwargs": {"enable_thinking": False}}
    req = urllib.request.Request(BASE + "/v1/chat/completions",
                                 json.dumps(body).encode(), H)
    with urllib.request.urlopen(req, timeout=900) as r:
        return json.load(r)["choices"][0]["message"]["content"]
r1 = ask(big)
r2 = ask(big)  # replay from the deep landing
print(f"cold answer:   {r1[:60]!r}", flush=True)
print(f"replay answer: {r2[:60]!r}", flush=True)
print(f"secret in cold={secret in r1} in replay={secret in r2} identical={r1 == r2}", flush=True)
