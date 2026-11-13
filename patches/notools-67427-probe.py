#!/usr/bin/env python3
"""#67427 port gate: drive the REAL serving path (ParserManager-composed
DelegatingParser). No-tools requests must keep tool-call markup as text;
tools requests must keep extracting calls (XML and hermes)."""
import sys
from types import SimpleNamespace
from transformers import AutoTokenizer

from vllm.entrypoints.generate.base.protocol import FunctionDefinition
from vllm.entrypoints.openai.chat_completion.protocol import (
    ChatCompletionToolsParam,
)
from vllm.parser.parser_manager import ParserManager

TOK = AutoTokenizer.from_pretrained("/models/qwen38fn-awq-g32")
TOOLS = [ChatCompletionToolsParam(
    type="function",
    function=FunctionDefinition(name="calc", description="calculate",
        parameters={"type": "object", "properties": {"x": {"type": "number"}}}),
)]

cls = ParserManager.get_parser(
    tool_parser_name="qwen3_xml", reasoning_parser_name="qwen3",
    enable_auto_tools=True, tokenizer=TOK)
assert cls is not None, "parser composition failed"
print(f"composed: {cls.__name__} (engine path)")

REAL = "<tool_call>\n<function=calc>\n<parameter=x>1</parameter>\n</function>\n</tool_call>"
HERMES = '<tool_call>\n{"name": "calc", "arguments": {"x": 1}}\n</tool_call>'
QUOTED = "Вот как выглядит вызов:\n" + REAL + "\nКонец примера."

def make_parser(tools):
    return cls(TOK, tools, chat_template_kwargs={"enable_thinking": False})

def nonstream(text, tools):
    req = SimpleNamespace(
        tools=tools,
        tool_choice="auto" if tools else "none",
        include_reasoning=True)
    p = make_parser(tools)
    reasoning, content, calls = p.parse(text, req, enable_auto_tools=True)
    return content, [c.name for c in (calls or [])]

def stream(text, tools):
    req = SimpleNamespace(
        tools=tools,
        tool_choice="auto" if tools else "none",
        include_reasoning=True)
    p = make_parser(tools)
    chunks = [text[i:i+7] for i in range(0, len(text), 7)]
    parts, names = [], []
    for i, ch in enumerate(chunks):
        dm = p.parse_delta(ch, [], req, None, finished=(i == len(chunks)-1))
        if dm is None: continue
        if dm.content: parts.append(dm.content)
        for tc in dm.tool_calls or []:
            if tc.function and tc.function.name: names.append(tc.function.name)
    return "".join(parts), names

CASES = [
    # (name, text, tools, want_content_has, want_calls)
    ("no-tools-real-xml",  REAL,    None, "<tool_call>", []),
    ("no-tools-hermes",    HERMES,  None, '"name": "calc"', []),
    ("no-tools-quoted",    QUOTED,  None, "<function=calc>", []),
    ("tools-real-xml",     REAL,    TOOLS, None, ["calc"]),
    ("tools-hermes",       HERMES,  TOOLS, None, ["calc"]),
]
fails = 0
for name, text, tools, want_c, want_calls in CASES:
    c_ns, calls_ns = nonstream(text, tools)
    c_st, calls_st = stream(text, tools)
    ok = (want_c is None or (c_ns and want_c in c_ns)) and calls_ns == want_calls \
         and (want_c is None or (c_st and want_c in c_st)) and calls_st == want_calls
    status = "PASS" if ok else "FAIL"
    if not ok: fails += 1
    print(f"{status} {name}: ns_calls={calls_ns} st_calls={calls_st} "
          f"ns_content={str(c_ns)[:70]!r} st_content={str(c_st)[:70]!r}")
print("ALL GREEN" if fails == 0 else f"{fails} FAILURES")
sys.exit(1 if fails else 0)
