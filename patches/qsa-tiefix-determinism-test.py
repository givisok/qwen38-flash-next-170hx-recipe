"""Determinism check after the QSA top-k tie fix: identical greedy requests
must return identical completions. Three profiles: plain RU, code, and a
hermes-shaped multi-turn + tools prompt (the old garbage-trigger form)."""
import json
import time
import urllib.request

URL = "http://localhost:8080/v1/chat/completions"

TOOLS = [
    {
        "type": "function",
        "function": {
            "name": "get_weather",
            "description": "Get current weather for a city",
            "parameters": {
                "type": "object",
                "properties": {
                    "city": {"type": "string"},
                    "units": {"type": "string", "enum": ["c", "f"]},
                },
                "required": ["city"],
            },
        },
    }
]

PROFILES = [
    ("ru_plain", None, [], "Объясни в двух абзацах, почему небо голубое, а закат красный.", 128),
    ("code", None, [], "Напиши на Python функцию быстрой сортировки с подробными комментариями.", 160),
    (
        "hermes_tools",
        "Ты — полезный ассистент. Отвечай по-русски, кратко и по делу.",
        TOOLS,
        [
            {"role": "user", "content": "Какая погода в Москве?"},
            {
                "role": "assistant",
                "content": None,
                "tool_calls": [
                    {
                        "id": "call_1",
                        "type": "function",
                        "function": {"name": "get_weather", "arguments": '{"city": "Москва", "units": "c"}'},
                    }
                ],
            },
            {"role": "tool", "tool_call_id": "call_1", "content": '{"temp_c": 14, "condition": "пасмурно", "wind_mps": 3}'},
            {"role": "user", "content": "Спасибо. А теперь своими словами: почему пасмурная погода часто сопровождается ветром? Ответь одним абзацем."},
        ],
        200,
    ),
]

REPEATS = 8


def call(system, tools, messages, max_tokens):
    if isinstance(messages, str):
        msgs = ([{"role": "system", "content": system}] if system else []) + [
            {"role": "user", "content": messages}
        ]
    else:
        msgs = ([{"role": "system", "content": system}] if system else []) + messages
    body = {
        "model": "qwen3.8",
        "messages": msgs,
        "temperature": 0,
        "max_tokens": max_tokens,
    }
    if tools:
        body["tools"] = tools
    req = urllib.request.Request(
        URL, data=json.dumps(body).encode(), headers={"Content-Type": "application/json"}
    )
    t0 = time.time()
    with urllib.request.urlopen(req, timeout=300) as r:
        out = json.loads(r.read())
    dt = time.time() - t0
    text = out["choices"][0]["message"].get("content") or ""
    usage = out.get("usage", {})
    return text, dt, usage


fails = 0
for name, system, tools, prompt, mt in PROFILES:
    outs = []
    t0 = time.time()
    for _ in range(REPEATS):
        text, _, _ = call(system, tools, prompt, mt)
        outs.append(text)
    distinct = len(set(outs))
    ok = distinct == 1
    fails += 0 if ok else 1
    status = "PASS" if ok else "FAIL"
    print(f"[{status}] {name}: {REPEATS} повторов -> {distinct} различных ответов ({time.time()-t0:.1f}s всего)")
    if not ok:
        for i, o in enumerate(outs):
            print(f"  --- run {i} head ---")
            print("  " + o[:200].replace("\n", "\n  "))
    else:
        print("  head:", outs[0][:150].replace("\n", " "))

print()
print("RESULT:", "ALL DETERMINISTIC" if fails == 0 else f"{fails} PROFILES NON-DETERMINISTIC")
