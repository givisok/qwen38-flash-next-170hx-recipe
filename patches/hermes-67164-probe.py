#!/usr/bin/env python3
"""Hermes-JSON body inside <tool_call> — hole probe for our qwen3 engine parser.

Upstream #67164/#67157: Qwen3 (base/uncensored finetunes) may emit
  <tool_call>{"name": ..., "arguments": ...}</tool_call>
instead of the trained XML form. Our stack: does the call survive?
CPU stand, thinking-off lane, non-stream + 7-char streaming.
"""
import sys
from types import SimpleNamespace

from transformers import AutoTokenizer

from vllm.entrypoints.generate.base.protocol import FunctionDefinition
from vllm.entrypoints.openai.chat_completion.protocol import (
    ChatCompletionToolsParam,
)
from vllm.parser.qwen3 import Qwen3Parser

TOK = AutoTokenizer.from_pretrained("/models/qwen38fn-awq-g32")

TOOLS = [
    ChatCompletionToolsParam(
        type="function",
        function=FunctionDefinition(
            name="calc",
            description="calculate",
            parameters={"type": "object", "properties": {"x": {"type": "number"}}},
        ),
    )
]
REQ = SimpleNamespace(tools=TOOLS, tool_choice="auto", include_reasoning=True)

REAL = "<tool_call>\n<function=calc>\n<parameter=x>1</parameter>\n</function>\n</tool_call>"
HERMES = '<tool_call>\n{"name": "calc", "arguments": {"x": 1}}\n</tool_call>'
HERMES_PARAMS = '<tool_call>\n{"name": "calc", "parameters": {"x": 1}}\n</tool_call>'
HERMES_TEXT = 'Сделаю расчёт.\n<tool_call>\n{"name": "calc", "arguments": {"x": 1}}\n</tool_call>'
HERMES_NOT_OFFERED = '<tool_call>\n{"name": "nukes", "arguments": {"x": 1}}\n</tool_call>'
HERMES_FENCED = "Пример разметки:\n```json\n" + HERMES + "\n```\nКонец примера."
HERMES_STR_ARGS = '<tool_call>\n{"name": "calc", "arguments": "{\\"x\\": 1}"}\n</tool_call>'


def new_parser():
    return Qwen3Parser(
        tokenizer=TOK,
        tools=TOOLS,
        chat_template_kwargs={"enable_thinking": False},
    )


def run_nonstream(text):
    p = new_parser()
    r = p.extract_tool_calls(text, REQ)
    return r.content or "", [
        (c.function.name, c.function.arguments) for c in (r.tool_calls or [])
    ]


def run_stream(text):
    p = new_parser()
    chunks = [text[i : i + 7] for i in range(0, len(text), 7)]
    calls, content_parts = [], []
    for i, ch in enumerate(chunks):
        dm = p.parse_delta(ch, [], REQ, None, finished=(i == len(chunks) - 1))
        if dm is None:
            continue
        if dm.content:
            content_parts.append(dm.content)
        for tc in dm.tool_calls or []:
            if tc.function and tc.function.name:
                calls.append((tc.function.name, getattr(tc.function, "arguments", None)))
    return "".join(content_parts), calls


CASES = [
    # (name, text, want_ns, want_st, want_name)
    ("real-xml-call", REAL, 1, 1, "calc"),
    ("hermes-full", HERMES, 1, 1, "calc"),
    ("hermes-parameters-key", HERMES_PARAMS, 1, 1, "calc"),
    ("hermes-with-text", HERMES_TEXT, 1, 1, "calc"),
    ("hermes-str-args", HERMES_STR_ARGS, 1, 1, "calc"),
    ("hermes-not-offered", HERMES_NOT_OFFERED, 0, 0, None),
    ("hermes-fenced", HERMES_FENCED, 0, 0, None),
]

fails = 0
for name, text, exp_ns, exp_st, want_name in CASES:
    c_ns, calls_ns = run_nonstream(text)
    c_st, calls_st = run_stream(text)
    ok_ns = len(calls_ns) == exp_ns and (want_name is None or any(n == want_name for n, _ in calls_ns))
    ok_st = len(calls_st) == exp_st and (want_name is None or any(n == want_name for n, _ in calls_st))
    status = "PASS" if (ok_ns and ok_st) else "HOLE" if exp_ns else "FAIL"
    if not (ok_ns and ok_st):
        fails += 1
    print(
        f"{status:4} {name}: ns={len(calls_ns)}/{exp_ns} st={len(calls_st)}/{exp_st} "
        f"ns_calls={calls_ns} st_calls={calls_st}"
    )
    print(f"     ns_content={c_ns[:80]!r} st_content={c_st[:80]!r}")

print("ALL GREEN" if fails == 0 else f"{fails} FAILURES")
sys.exit(1 if fails else 0)
