import json, time, threading, urllib.request, sys

BASE = "http://localhost:8080"
KEY = "***REDACTED-API-KEY***"
H = {"Content-Type": "application/json", "Authorization": "Bearer " + KEY}

PROMPTS = [
    "Write a Python function computing the nth Fibonacci with memoization.",
    "Write a Rust function reversing a linked list iteratively.",
    "Refactor this SQL: SELECT * FROM users WHERE id IN (SELECT user_id FROM orders);",
    "Write a Python LRU cache class with get/put O(1).",
    "Write a Go function merging two sorted channels.",
    "Fix the off-by-one: for i in range(1, len(a)): if a[i] > a[i-1]: m = a[i].",
    "Write a TypeScript debounce function with cancel.",
    "Write a C function counting set bits (popcount).",
    "Write a Python generator yielding prime numbers.",
    "Write a bash one-liner finding the 10 largest files under a dir.",
    "Write a Java singleton with double-checked locking.",
    "Write a regex validating ISO dates 1583..9999, explain it.",
    "Write a JS deep-clone without JSON.",
    "Write a Kotlin extension summing a list of money amounts.",
    "Write a Python async retry decorator with exponential backoff.",
    "Write a SQL window query ranking sales per region per month.",
    "Write a Rust iterator chain computing word frequencies.",
    "Write a Python function parsing RFC3339 without libs.",
    "Write a Zig function swapping two ints safely.",
    "Write a Ruby monkey-patch adding .to_shorthand on Integer (bytes).",
    "Write a Python AVL insertion, balanced.",
    "Write a JS function grouping array items by a key callback.",
    "Write a Haskell quicksort in 5 lines, explain complexity.",
    "Write a Python function topological-sorting a DAG.",
    "Write a C++ RAII file handle class.",
    "Write a SQL recursive CTE for org-chart depth.",
    "Write a Python thread-safe bounded queue from scratch.",
    "Write a Lua memoize wrapper.",
    "Write a Perl one-liner reversing words per line.",
    "Write a Python function detecting a cycle in a linked list.",
    "Write a Scala tail-recursive factorial.",
    "Write a Raku class with operator overloading for vectors.",
]

def one(p, idx, out):
    body = {"model": "qwen3.8", "prompt": p, "max_tokens": 400,
            "temperature": 0, "stream": True}
    req = urllib.request.Request(BASE + "/v1/completions",
                                 data=json.dumps(body).encode(), headers=H)
    t0 = time.time(); dec = 0.0; ok = False
    try:
        with urllib.request.urlopen(req, timeout=1800) as r:
            for line in r:
                if not line.startswith(b"data: "): continue
                payload = line[6:].strip()
                if payload == b"[DONE]": break
                try: d = json.loads(payload)
                except Exception: continue
                ch = d.get("choices")
                if not ch: continue
                delta = ch[0].get("text") or ""
                if delta:
                    dec += len(delta); ok = True
        out[idx] = (ok, dec, time.time() - t0)
    except Exception as e:
        out[idx] = (False, 0.0, time.time() - t0)
        print("ERR", repr(e)[:120])

def run(n):
    prompts = [PROMPTS[i % len(PROMPTS)] + f" (variant {i})" for i in range(n)]
    out = [None] * n
    th = [threading.Thread(target=one, args=(p, i, out)) for i, p in enumerate(prompts)]
    t0 = time.time()
    for t in th: t.start()
    for t in th: t.join()
    wall = time.time() - t0
    total = sum(o[1] for o in out if o)
    oks = sum(1 for o in out if o and o[0])
    print(f"N={n}: wall {wall:.1f}s | decode aggregate {total/wall:.0f} tok/s | ok {oks}/{n}")

if __name__ == "__main__":
    for n in [int(x) for x in (sys.argv[1:] or ["8", "16", "32"])]:
        run(n)
