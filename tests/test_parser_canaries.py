#!/usr/bin/env python3
"""Canary battery for the qwen3 engine tool parser (CPU stand).

Runs against whatever tree is on PYTHONPATH. Exit code 0 = all green;
prints PASS/FAIL per case. Cases marked [phantom] must produce NO calls.
"""
import json
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
            parameters={
                "type": "object",
                "properties": {"x": {"type": "number"}},
            },
        ),
    )
]
REQ = SimpleNamespace(tools=TOOLS, tool_choice="auto", include_reasoning=True)

REAL = "<tool_call>\n<function=calc>\n<parameter=x>1</parameter>\n</function>\n</tool_call>"
BARE = "<function=calc>\n<parameter=x>2</parameter>\n</function>"
FENCED = "Here is how it looks:\n```xml\n" + REAL.replace("\n", "\n") + "\n```\nDone."
NOT_OFFERED = "<tool_call>\n<function=nukes>\n<parameter=x>3</parameter>\n</function>\n</tool_call>"
FENCE_THEN_REAL = "Example:\n```\n" + BARE + "\n```\nNow really call it:\n" + REAL
TWO = REAL + "\n" + REAL
PLAIN = "Обычный текст без разметки, 2+2=4."


def new_parser():
    # thinking off => CONTENT start: the lane where the CONTENT-side
    # fallback and fence phantoms live (same as the API canaries)
    return Qwen3Parser(
        tokenizer=TOK,
        tools=TOOLS,
        chat_template_kwargs={"enable_thinking": False},
    )


def run_nonstream(text):
    p = new_parser()
    r = p.extract_tool_calls(text, REQ)
    return r.content or "", list(r.tool_calls)


def run_stream(text):
    p = new_parser()
    req = REQ
    chunks = [text[i : i + 7] for i in range(0, len(text), 7)]
    calls, content_parts = [], []
    for i, ch in enumerate(chunks):
        dm = p.parse_delta(
            ch, [], req, None, finished=(i == len(chunks) - 1)
        )
        if dm is None:
            continue
        if dm.content:
            content_parts.append(dm.content)
        if dm.tool_calls:
            for tc in dm.tool_calls:
                if tc.function and tc.function.name:
                    calls.append(tc.function.name)
    return "".join(content_parts), calls


CASES = [
    # (name, text, expect_calls_nonstream, expect_calls_stream)
    ("real-call", REAL, 1, 1),
    ("plain-text", PLAIN, 0, 0),
    ("phantom-bare-function", BARE, 0, 0),
    ("phantom-fenced", FENCED, 0, 0),
    ("phantom-not-offered-name", NOT_OFFERED, 0, 0),
    ("fence-then-real-call", FENCE_THEN_REAL, 1, 1),
    ("two-real-calls", TWO, 2, 2),
]

fails = 0
for name, text, exp_ns, exp_st in CASES:
    content_ns, calls_ns = run_nonstream(text)
    content_st, calls_st = run_stream(text)
    ok_ns = len(calls_ns) == exp_ns
    ok_st = len(calls_st) == exp_st
    status = "PASS" if (ok_ns and ok_st) else "FAIL"
    if status == "FAIL":
        fails += 1
    print(
        f"{status} {name}: nonstream={len(calls_ns)}/{exp_ns} "
        f"stream={len(calls_st)}/{exp_st} "
        f"ns_calls={[c.function.name for c in calls_ns] if calls_ns else []} "
        f"st_calls={calls_st}"
    )

print(f"{'ALL GREEN' if fails == 0 else f'{fails} FAILURES'}")
sys.exit(1 if fails else 0)
