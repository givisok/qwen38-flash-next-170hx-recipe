#!/usr/bin/env python3
# Probe: truncated tool-call behavior (upstream #67594 class) on the live
# API. Cut vs full, non-streaming, thinking off, temp 0. Run ON the CT:
#   . /root/.qwen38fn_env && python3 /root/qwen38fn/tools/trunc_probe.py
# 2026-11-14 result (our prod, fork @ f010ce5415):
#   cut10 | finish=length | content='' | tool_calls=[] | 10 tokens burned
#   full  | finish=tool_calls | call calc fires
# => loss exists but is VISIBLE (finish=length); upstream fix (#67594)
# touches only pythonic/llama4/olmo3 parsers, not qwen3_xml. Shelf.
import json, os, urllib.request, urllib.error

B = "http://localhost:8080/v1/chat/completions"
K = os.environ["QWEN38FN_API_KEY"]
tools = [{"type": "function", "function": {"name": "calc", "description": "Калькулятор",
          "parameters": {"type": "object", "properties": {"x": {"type": "number"}},
                         "required": ["x"]}}}]

def go(mt, tag):
    body = {"model": "qwen3.8",
            "messages": [{"role": "user",
                          "content": "Вызови инструмент calc с x=12345. Только вызов инструмента, без текста."}],
            "tools": tools, "max_tokens": mt, "temperature": 0,
            "chat_template_kwargs": {"thinking": False}}
    req = urllib.request.Request(B, data=json.dumps(body).encode(),
                                 headers={"Authorization": "Bearer " + K,
                                          "Content-Type": "application/json"})
    try:
        with urllib.request.urlopen(req, timeout=180) as r:
            d = json.load(r)
    except urllib.error.HTTPError as e:
        print(tag, "HTTP", e.code, e.read()[:200].decode(errors="replace")); return
    ch = d["choices"][0]; m = ch["message"]
    tcs = json.dumps([{"fn": tc["function"]["name"], "args": tc["function"]["arguments"]}
                      for tc in (m.get("tool_calls") or [])], ensure_ascii=False)
    print(tag, "| finish:", ch["finish_reason"], "| content:", repr((m.get("content") or "")[:150]))
    print(tag, "| tool_calls:", tcs[:200], "| completion_tokens:", d["usage"]["completion_tokens"])

go(10, "cut10")
go(200, "full ")
