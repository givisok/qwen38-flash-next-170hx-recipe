"""One A/B arm: determinism battery + code-heavy decode battery + spec metrics.

usage: python3 /root/ab_measure.py <tag>
"""
import concurrent.futures as cf
import json
import re
import subprocess
import sys
import time
import urllib.request

URL = "http://localhost:8080/v1/chat/completions"
CODE_PROMPTS = [
    "Напиши на Python функцию парсинга CSV без библиотек: кавычки, экранирование, многострочные поля. Полный код с докстрингами и примером.",
    "Реализуй на Python LRU-кэш с таймаутом TTL, потокобезопасный, с подробными комментариями. Полный код.",
    "Напиши на Python скрипт: обход дерева каталогов, подсчёт строк по расширениям, вывод топ-10. Только stdlib. Полный код с docstring.",
    "Реализуй быструю сортировку и сортировку слиянием на Python, сравни сложность в комментариях, добавь тесты. Полный код.",
]


def chat(prompt, max_tokens=600, temperature=0.0, tools=None, msgs=None):
    if msgs is None:
        msgs = [{"role": "user", "content": prompt}]
    body = {
        "model": "qwen3.8",
        "messages": msgs,
        "temperature": temperature,
        "max_tokens": max_tokens,
    }
    if tools:
        body["tools"] = tools
    req = urllib.request.Request(
        URL, data=json.dumps(body).encode(), headers={"Content-Type": "application/json"}
    )
    t0 = time.time()
    with urllib.request.urlopen(req, timeout=600) as r:
        out = json.loads(r.read())
    dt = time.time() - t0
    text = out["choices"][0]["message"].get("content") or ""
    ctok = out.get("usage", {}).get("completion_tokens", 0)
    return text, dt, ctok


def metrics():
    with urllib.request.urlopen("http://localhost:8080/metrics", timeout=30) as r:
        txt = r.read().decode()

    def g(name):
        m = re.search(rf"^{name}{{[^}}]*}} ([\d.e+]+)", txt, re.M)
        return float(m.group(1)) if m else 0.0

    drafts = g("vllm:spec_decode_num_drafts_total")
    dtok = g("vllm:spec_decode_num_draft_tokens_total")
    acc = g("vllm:spec_decode_num_accepted_tokens_total")
    pos = [g(f'vllm:spec_decode_num_accepted_tokens_per_pos_total{{engine="0",model_name="qwen3.8",position="{i}"}}') for i in (0, 1, 2)]
    return {
        "accept_rate": acc / dtok if dtok else 0.0,
        "mean_acc_per_draft": acc / drafts if drafts else 0.0,
        "pos_rate": [p / drafts if drafts else 0.0 for p in pos],
    }


def main():
    tag = sys.argv[1]
    print(f"=== ARM {tag} ===")

    # 1) determinism battery
    r = subprocess.run(
        ["python3", "/root/determinism_test.py"], capture_output=True, text=True, timeout=3600
    )
    out = r.stdout + r.stderr
    passes = out.count("[PASS]")
    fails = out.count("[FAIL]")
    print(f"determinism: pass={passes} fail={fails} ({out.strip().splitlines()[-1]})")

    # 2) code battery, 4 parallel, greedy
    t0 = time.time()
    with cf.ThreadPoolExecutor(max_workers=4) as ex:
        futs = [ex.submit(chat, p) for p in CODE_PROMPTS]
        results = [f.result() for f in futs]
    span = time.time() - t0
    tot_tok = sum(c for _, _, c in results)
    per = [c / dt for _, dt, c in results]
    print(f"code battery: agg {tot_tok/span:.1f} tok/s, per-req avg {sum(per)/len(per):.1f} tok/s")
    for i, (txt, dt, c) in enumerate(results):
        ok = "```" in txt and "def " in txt
        print(f"  code{i}: {c} tok /{dt:.1f}s fenced={ok}")
    with open(f"/root/ab-{tag}-code.json", "w") as f:
        json.dump([t for t, _, _ in results], f, ensure_ascii=False, indent=1)

    # 3) spec metrics
    print("spec:", json.dumps(metrics(), ensure_ascii=False))


if __name__ == "__main__":
    main()
