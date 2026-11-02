# 1M-context multi-needle retrieval gate: one ~1.0M-token record wall,
# needles at 6 depths (5/20/40/60/80/95%), each asked 3 times with different
# question phrasings (first query = cold prefill; the rest must ride the
# prefix cache on the shared document), plus one negative control.
import json, random, time, urllib.request
from collections import defaultdict

BASE = "http://localhost:8080"
import os
KEY = os.environ["QWEN38FN_API_KEY"]  # export QWEN38FN_API_KEY first
H = {"Content-Type": "application/json", "Accept": "text/event-stream",
     "Authorization": "Bearer " + KEY}
MODEL = "qwen3.8"

TARGET_TOKENS = 1_010_000  # leave ~38k headroom under max_model_len 1,048,576
CH_PER_TOK = 1.52025  # measured via /tokenize on this filler (1.763 was wrong)

rng = random.Random(1_000_000)
STATIONS = [("ALPHA", 0.05), ("BRAVO", 0.20), ("CHARLIE", 0.40),
            ("DELTA", 0.60), ("ECHO", 0.80), ("FOXTROT", 0.95)]
CODES = {name: "%06d" % rng.randrange(1_000_000) for name, _ in STATIONS}


def tokenize_len(text):
    req = urllib.request.Request(BASE + "/tokenize",
                                 data=json.dumps({"model": MODEL, "prompt": text}).encode(),
                                 headers=H)
    return len(json.load(urllib.request.urlopen(req, timeout=600))["tokens"])


def build_doc(ch_per_tok):
    total_chars = int(TARGET_TOKENS * ch_per_tok)
    rows, i, n = [], 0, 0
    while n < total_chars:
        w = ("rec %07d sku-%06x qty%03d bin %02d-%02d op%02x | " %
             (i, rng.getrandbits(24), rng.getrandbits(10) % 500,
              rng.getrandbits(7), rng.getrandbits(6), rng.getrandbits(8)))
        rows.append(w); n += len(w); i += 1
    filler = "".join(rows)
    out, prev = [], 0
    for name, frac in STATIONS:
        pos = int(len(filler) * frac)
        out.append(filler[prev:pos])
        out.append(f"\nNOTE: the secret access code for station {name} is "
                   f"{CODES[name]}. Remember this code. \n")
        prev = pos
    out.append(filler[prev:])
    return "".join(out)


DOC, DOC_TOK = None, 0
for _ in range(3):
    DOC = build_doc(CH_PER_TOK)
    DOC_TOK = tokenize_len(DOC)
    print(f"doc chars={len(DOC)} tokens={DOC_TOK} (target {TARGET_TOKENS})", flush=True)
    if abs(DOC_TOK - TARGET_TOKENS) <= 8000:
        break
    CH_PER_TOK = CH_PER_TOK * TARGET_TOKENS / DOC_TOK
else:
    raise SystemExit(f"calibration failed to converge: {DOC_TOK} tokens")
if DOC_TOK > 1_030_000:
    raise SystemExit(f"doc too big even after calibration: {DOC_TOK} tokens")
print(f"codes={CODES}", flush=True)

PHRASES = [
    "What is the secret access code for station {s} mentioned among the records? Answer with just the six-digit code.",
    "Reply with only the six-digit secret code for station {s} from the records above.",
    "In the records there is a secret access code for station {s}. What is it? Six digits only.",
]


def ask(question, max_tokens=32):
    body = {"model": MODEL,
            "messages": [{"role": "user", "content": DOC + "\n\n" + question}],
            "max_tokens": max_tokens, "temperature": 0, "top_p": 1,
            "stream": True, "stream_options": {"include_usage": True},
            "chat_template_kwargs": {"enable_thinking": False}}
    req = urllib.request.Request(BASE + "/v1/chat/completions",
                                 data=json.dumps(body).encode(), headers=H)
    t0 = time.time(); tfirst = None; parts = []; usage = {}
    with urllib.request.urlopen(req, timeout=1800) as r:
        for line in r:
            if not line.startswith(b"data: "):
                continue
            p = line[6:].strip()
            if p == b"[DONE]":
                break
            try:
                d = json.loads(p)
            except Exception:
                continue
            if d.get("usage"):
                usage = d["usage"]; continue
            ch = d.get("choices")
            if ch and (ch[0].get("delta", {}) or {}).get("content"):
                if tfirst is None:
                    tfirst = time.time()
                parts.append(ch[0]["delta"]["content"])
    return {"ans": "".join(parts), "total": time.time() - t0,
            "ttft": (tfirst - t0) if tfirst else 0.0, "usage": usage}


def one_query(label, question, check):
    for attempt in (1, 2):
        try:
            r = ask(question)
            break
        except Exception as e:
            detail = ""
            if hasattr(e, "read"):
                try:
                    detail = " | " + e.read().decode()[:200]
                except Exception:
                    pass
            print(f"  retry {label}: {type(e).__name__}: {e}{detail}", flush=True)
            time.sleep(10)
    else:
        print(f"{label}: ERROR (both attempts failed)", flush=True)
        return "ERROR"
    u = r["usage"] or {}
    cached = (u.get("prompt_tokens_details") or {}).get("cached_tokens", 0)
    ptok = u.get("prompt_tokens", 0)
    ans = r["ans"].strip()
    verdict = check(ans)
    print(f"{label}: {verdict} | ptok={ptok} cached={cached} | "
          f"ttft={r['ttft']:.1f}s total={r['total']:.1f}s "
          f"comp={u.get('completion_tokens', 0)} | {ans[:60]!r}", flush=True)
    return verdict


results = defaultdict(list)
for rnd, phrase in enumerate(PHRASES, 1):
    for name, frac in STATIONS:
        label = f"r{rnd} {name}@{int(frac*100)}%"
        v = one_query(label, phrase.format(s=name),
                      lambda a, n=name: "PASS" if CODES[n] in a else "FAIL")
        results[f"{name}@{int(frac*100)}%"].append(v)

v = one_query("ctrl ZULU (absent station)",
              PHRASES[0].format(s="ZULU"),
              lambda a: "OK-neg" if not any(c in a for c in CODES.values())
              else "LEAK")
results["ctrl-ZULU"].append(v)

print("\n=== SUMMARY (per depth, rounds in order) ===", flush=True)
for label in [f"{n}@{int(f*100)}%" for n, f in STATIONS] + ["ctrl-ZULU"]:
    print(f"{label}: {results[label]}", flush=True)
npass = sum(1 for vs in results.values() for x in vs if x == "PASS")
print(f"needle queries passed: {npass}/18", flush=True)
