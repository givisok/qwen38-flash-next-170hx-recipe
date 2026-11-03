# One body (~26k chars), three sends: publish -> replay -> replay.
# Wall-timestamps printed per send to join with engine [hit-diag]/[cache-diag].
import json, os, time, urllib.request, datetime

BASE = "http://localhost:8080"
KEY = os.environ["QWEN38FN_API_KEY"]
H = {"Content-Type": "application/json", "Authorization": "Bearer " + KEY}
RU = open("/root/qwen38fn/tools/ru_block.txt", encoding="utf-8").read()

body = "[диаг-глубина X43] " + (RU * 8)[:26000]

def send(content):
    b = {"model": "qwen3.8", "messages": [{"role": "user", "content": content}],
         "max_tokens": 8, "temperature": 0,
         "chat_template_kwargs": {"enable_thinking": False}}
    req = urllib.request.Request(BASE + "/v1/chat/completions", json.dumps(b).encode(), H)
    t0 = time.time()
    with urllib.request.urlopen(req, timeout=600) as r:
        j = json.load(r)
    u = j.get("usage", {}); det = u.get("prompt_tokens_details") or {}
    print(f"{datetime.datetime.now(datetime.timezone.utc).strftime('%H:%M:%S')} "
          f"utc | prompt {u.get('prompt_tokens')} | cached {det.get('cached_tokens')}", flush=True)

for i in range(3):
    send(body)
    time.sleep(2)
print("DONE", flush=True)
