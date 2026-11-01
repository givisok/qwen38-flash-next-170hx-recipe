# sparkDash DecodeBench protocol replica (MiaAI-Lab/sparkDash @ 3e6d4ef,
# server/collectors/DecodeBench.js + LlmStreaming.js), exact:
# - POST /v1/chat/completions, stream, include_usage
# - temperature 0, top_p 1, thinking off, max_tokens 400
# - one 32-token warmup request (warmup_noop task) before each wave
# - decodeTokens = completion_tokens - 1 (from usage)
# - per-request decodeTps = decodeTokens / (tLast - tFirst)  [content-chunk ts]
# - wave: median per-request decodeTps; aggregate = totalDecodeTokens /
#   (max(tLast) - min(tFirst))
# - 3 passes per level, medians reported (AlexBrandes protocol)
import json, statistics, sys, threading, time, urllib.request

BASE = "http://localhost:8080"
KEY = "***REDACTED-API-KEY***"
H = {"Content-Type": "application/json", "Accept": "text/event-stream",
     "Authorization": "Bearer " + KEY}

TAIL = (" Output only Python source. No comments, no docstrings, no markdown"
        " fences. Then add tests and the helpers this needs. Keep writing code.")

def code_task(name, spec):
    return f"{name}\n{spec}\n{TAIL}"

TASKS = [
    code_task("binary_search", "def binary_search(nums, target) -> int: index of target in a sorted list, or -1."),
    code_task("merge_sort", "def merge_sort(nums) -> list: stable sort of a list of ints, returning a new list."),
    code_task("lru_cache", "class LRUCache: get(key) and put(key, value) with a fixed capacity, evicting the least recently used."),
    code_task("token_bucket", "class TokenBucket: allow(n) consumes n tokens refilled at a fixed rate, else returns False."),
    code_task("ring_buffer", "class RingBuffer: push and pop over a fixed-capacity array, raising on overflow and underflow."),
    code_task("dijkstra", "def dijkstra(graph, src) -> dict: shortest path weights from src on a non-negative weighted graph."),
    code_task("edit_distance", "def edit_distance(a, b) -> int: Levensstein distance between two strings."),
    code_task("semver_cmp", "def semver_cmp(a, b) -> int: compare dotted numeric versions, negative if a < b."),
    code_task("url_parse", "def url_parse(url) -> dict: scheme, host, port, path, and query pairs. No extra libraries."),
    code_task("json_pointer", "def json_pointer(doc, pointer) -> object: follow an RFC 6901 pointer, or None if missing."),
    code_task("glob_match", "def glob_match(pattern, text) -> bool: * and ? wildcards, no character classes."),
    code_task("csv_parse", "def csv_parse(text) -> list: rows of fields, honoring double-quoted commas and escaped quotes."),
    code_task("rle", "def rle_encode(s) -> str and rle_decode(s) -> str: run-length encoding of single-byte runs."),
    code_task("top_k", "def top_k(nums, k) -> list: the k largest ints, unordered, using a bounded heap."),
    code_task("interval_merge", "def merge_intervals(spans) -> list: merge overlapping [start, end] pairs."),
    code_task("topo_sort", "def topo_sort(nodes, edges) -> list: a valid order, or None if the graph has a cycle."),
    code_task("bloom_filter", "class BloomFilter: add(item) and might_contain(item) with two hash functions over a bit array."),
    code_task("moving_average", "class MovingAverage: next(x) returns the mean of the last window values."),
    code_task("retry_backoff", "def backoff_delays(attempts, base_ms, cap_ms) -> list: exponential delays clipped at the cap."),
    code_task("base64_encode", "def b64_encode(data) -> str and b64_decode(text) -> bytes: standard base64, no libraries."),
    code_task("expr_eval", "def eval_expr(text) -> int: evaluate non-negative ints with + - * / and parentheses."),
    code_task("histogram_percentile", "def percentile(samples, p) -> float: nearest-rank percentile of a list of numbers."),
    code_task("redact_secrets", "def redact(text) -> str: replace AWS-looking keys and password= values with ***. Keep the rest."),
    code_task("chunk_text", "def chunk_text(text, size) -> list: split into chunks of at most size chars, breaking on spaces when possible."),
    code_task("route_match", "def route_match(pattern, path) -> dict or None: /users/:id style params."),
    code_task("crc32", "def crc32(data) -> int: IEEE CRC-32 of a bytes object."),
    code_task("fixed_window", "class FixedWindow: allow() is True up to limit events per window_s, else False."),
    code_task("diff_lines", "def diff_lines(a, b) -> list: line diff as equal/delete/insert ops using a simple LCS."),
    code_task("infix_postfix", "def infix_to_postfix(tokens) -> list: shunting-yard for + - * / and parentheses."),
    code_task("consistent_hash", "class ConsistentHash: add_node, remove_node, and get_node(key) on a ring of virtual nodes."),
    code_task("utf8_decode", "def utf8_decode(data) -> str: decode UTF-8 bytes, replacing invalid sequences with U+FFFD."),
    code_task("dependency_closure", "def closure(root, deps) -> list: packages reachable from root, each name once, in visit order."),
]
WARMUP = code_task("warmup_noop", "def warmup_noop(x): return x unchanged.")
PROSE = ("Write a detailed step-by-step explanation of how a hash map works, "
         "including collision handling, resizing, and time complexity. Be thorough.")
STRUCTURED = "Count from 1 to 200. Output only the numbers, separated by spaces. No other text."

def one(prompt, max_tokens, out, idx):
    body = {"model": "qwen3.8", "messages": [{"role": "user", "content": prompt}],
            "max_tokens": max_tokens, "temperature": 0, "top_p": 1,
            "stream": True, "stream_options": {"include_usage": True},
            "chat_template_kwargs": {"enable_thinking": False}}
    req = urllib.request.Request(BASE + "/v1/chat/completions",
                                 data=json.dumps(body).encode(), headers=H)
    t0 = time.time(); t_first = t_last = None; comp = 0
    try:
        with urllib.request.urlopen(req, timeout=600) as r:
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
        ok = comp > 0 and t_first and t_last and t_last > t_first
        out[idx] = {
            "ok": bool(ok), "completion_tokens": comp,
            "decode_tokens": max(0, comp - 1) if comp else 0,
            "t0": t0, "t_first": t_first, "t_last": t_last,
            "decode_tps": ((comp - 1) / (t_last - t_first)) if ok else 0.0,
            "ttft_ms": (t_first - t0) * 1000 if t_first else None,
        }
    except Exception as e:
        out[idx] = {"ok": False, "err": repr(e)[:100], "completion_tokens": 0,
                    "decode_tokens": 0, "t0": t0, "t_first": None, "t_last": None,
                    "decode_tps": 0.0, "ttft_ms": None}

def warmup():
    o = [None]
    one(WARMUP, 32, o, 0)
    return o[0]["ok"]

def wave(n, ptype):
    if ptype == "code":
        prompts = TASKS[:n] if n <= len(TASKS) else [TASKS[i % len(TASKS)] for i in range(n)]
    elif ptype == "prose":
        prompts = [PROSE] * n
    else:
        prompts = [STRUCTURED] * n
    warmup()
    out = [None] * n
    th = [threading.Thread(target=one, args=(p, 400, out, i)) for i, p in enumerate(prompts)]
    for t in th:
        t.start()
    for t in th:
        t.join()
    ok = [o for o in out if o and o["ok"]]
    if not ok:
        print(f"  WAVE FAILED ({sum(1 for o in out if o and o['ok'])}/{n} ok)")
        return None
    med = statistics.median(o["decode_tps"] for o in ok)
    tot = sum(o["decode_tokens"] for o in ok)
    win = max(o["t_last"] for o in ok) - min(o["t_first"] for o in ok)
    agg = tot / win if win > 0 else 0.0
    medttft = statistics.median(o["ttft_ms"] for o in ok if o["ttft_ms"] is not None)
    return {"median_tps": med, "aggregate_tps": agg, "median_ttft_ms": medttft,
            "per_req_tps": [o["decode_tps"] for o in ok], "n_ok": len(ok)}

def main():
    levels = [int(x) for x in (sys.argv[1:] or ["1", "8", "16", "32"])]
    passes = 3
    for ptype in ("code", "prose", "structured"):
        print(f"== {ptype} ==")
        for n in levels:
            res = []
            for p in range(passes):
                w = wave(n, ptype)
                if w:
                    res.append(w)
                time.sleep(2)
            if res:
                mmed = statistics.median(w["median_tps"] for w in res)
                magg = statistics.median(w["aggregate_tps"] for w in res)
                mttft = statistics.median(w["median_ttft_ms"] for w in res)
                print(f"  N={n}: per-req median {mmed:.1f} tok/s | aggregate {magg:.1f} tok/s | TTFT {mttft:.0f} ms | passes {len(res)}/{passes}")

if __name__ == "__main__":
    main()
