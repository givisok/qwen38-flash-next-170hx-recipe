#!/usr/bin/env python3
"""Prefill (PP) + generation (TG) benchmark for qwen3.8."""
import json, random, time, urllib.request

URL = "http://127.0.0.1:8080/v1/chat/completions"
WORDS = ("система данных модуль процесс анализ результат параметр функция структура "
         "алгоритм уровень ресурс поток объект контроль метод цикл версия проект "
         "расчёт измерение сигнал устройство платформа").split()

def gen_prompt(n_par):
    random.seed(1234)
    return "\n".join(
        f"Запись {i}: " + " ".join(random.choices(WORDS, k=random.randint(60, 90))) + "."
        for i in range(n_par))

def call(prompt, max_tokens, marker):
    body = json.dumps({"model": "qwen3.8", "max_tokens": max_tokens,
                       "temperature": 0,
                       "messages": [{"role": "user",
                                     "content": prompt + f" Маркер {marker}."}]}).encode()
    req = urllib.request.Request(URL, data=body,
                                 headers={"Content-Type": "application/json"})
    t0 = time.time()
    with urllib.request.urlopen(req, timeout=600) as r:
        d = json.loads(r.read())
    return time.time() - t0, d

print("=== PP: prompt processing (prefill), max_tokens=1 ===")
for n_par, label in [(25, "~2K"), (100, "~8K"), (200, "~16K"), (400, "~32K"), (800, "~64K")]:
    p = gen_prompt(n_par)
    for try_i, kind in [(0, "cold"), (1, "warm")]:
        w, d = call(p, 1, f"{n_par}-{try_i}-{time.time_ns()//1000}")
        pt = d.get("usage", {}).get("prompt_tokens", 0)
        print(f"{label:>6} ({pt:>6} tok): {w:6.2f}s = {pt / w if w else 0:7.1f} tok/s [{kind}]")

print("\n=== TG: text generation (decode), 900 tok out ===")
p = gen_prompt(3)
for i in range(3):
    w, d = call(p, 900, f"tg-{i}-{time.time_ns()//1000}")
    ct = d.get("usage", {}).get("completion_tokens", 0)
    print(f"run{i + 1}: {ct} tok in {w:.1f}s = {ct / w:.1f} tok/s")
