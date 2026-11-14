"""Shared fixtures for the benchmark tests: on-disk runs and a fake agent.

Nothing here calls a real LLM or the network. The only fake is the chat model
(the LLM boundary); spans come from the real OpenInference instrumentation.
"""

from __future__ import annotations

import re
import zlib
from pathlib import Path

from langchain_core.language_models.fake_chat_models import FakeMessagesListChatModel
from langchain_core.messages import AIMessage
from langchain_core.tools import tool
from langgraph.prebuilt import create_react_agent

from evals.benchmark.judge import JudgeVerdict
from evals.benchmark.runs import QUESTIONS_DIR, RUN_FILE, write_model
from evals.benchmark.schemas import Pricing, QuestionResult, RunMeta, ToolCall

QUESTION_SET_HASH = "hash-of-the-fixture-question-set"

# (id, ticker, category)
FIXTURE_QUESTIONS = [
    ("q-risk", "AAPL", "filing_lookup"),
    ("q-multi", "NVDA", "multi_step"),
    ("q-8k", "JPM", "material_events"),
    ("q-tech", "AAPL", "technicals"),
    ("q-news", "NVDA", "market_news"),
]

PRICED = Pricing(input_price_per_mtok=2.0, output_price_per_mtok=10.0, prices_as_of="2026-01-15")
UNPRICED = Pricing()


def make_result(qid: str, ticker: str, category: str, answer: str | None, **overrides) -> QuestionResult:
    fields = dict(
        id=qid,
        ticker=ticker,
        category=category,
        question=f"What about {ticker} ({qid})?",
        status="ok" if answer else "failed",
        answer=answer,
        llm_calls=3,
        input_tokens=1000,
        output_tokens=200,
        total_tokens=1200,
        latency_s=10.0,
        llm_time_s=6.0,
        tool_calls=[ToolCall(name="get_risk_factors", duration_s=1.5)],
    )
    fields.update(overrides)
    return QuestionResult(**fields)


def write_run(
    root: Path,
    name: str,
    *,
    architecture: str = "arch-x",
    model_id: str = "model-x",
    pricing: Pricing = PRICED,
    answers: dict[str, str | None] | None = None,
    question_set_hash: str = QUESTION_SET_HASH,
    tavily_enabled: bool = True,
    **result_overrides,
) -> Path:
    """Write a run directory with the real schema; returns its path."""
    run_dir = root / name
    ids = [q[0] for q in FIXTURE_QUESTIONS]
    meta = RunMeta(
        run_id=f"run-{name}",
        architecture=architecture,
        model_id=model_id,
        git_sha="abc1234",
        git_dirty=False,
        date="2026-02-01T10:00:00+00:00",
        question_set_hash=question_set_hash,
        question_ids=ids,
        tavily_enabled=tavily_enabled,
        pricing=pricing,
    )
    write_model(run_dir / RUN_FILE, meta)
    for qid, ticker, category in FIXTURE_QUESTIONS:
        answer = (answers or {}).get(qid, f"{name} answer for {qid}")
        write_model(
            run_dir / QUESTIONS_DIR / f"{qid}.json",
            make_result(qid, ticker, category, answer, **result_overrides),
        )
    return run_dir


@tool
def lookup_fact(topic: str) -> str:
    """Look up a fact about a topic."""
    return f"fact about {topic}"


class _ScriptedChatModel(FakeMessagesListChatModel):
    """Fake chat model that tolerates `bind_tools` (as `create_react_agent` needs)."""

    def bind_tools(self, tools, **kwargs):
        return self


class FakeAgent:
    """A real LangGraph ReAct agent on a scripted chat model, with the stream() event contract.

    One tool call then an answer, with token usage on both LLM calls, so the
    instrumentation emits genuine LLM and TOOL spans.
    """

    def __init__(self, answer: str = "The answer is 42.", *, usage: bool = True):
        usage_1 = {"input_tokens": 100, "output_tokens": 10, "total_tokens": 110} if usage else None
        usage_2 = {"input_tokens": 150, "output_tokens": 20, "total_tokens": 170} if usage else None
        llm = _ScriptedChatModel(
            responses=[
                AIMessage(
                    content="",
                    tool_calls=[{"name": "lookup_fact", "args": {"topic": "x"}, "id": "call_1"}],
                    usage_metadata=usage_1,
                ),
                AIMessage(content=answer, usage_metadata=usage_2),
            ]
        )
        self._agent = create_react_agent(llm, [lookup_fact])

    async def stream(self, inputs):
        yield {"type": "node", "node": "react_agent", "message": "thinking"}
        result = await self._agent.ainvoke(inputs)
        yield {"type": "response", "message": result["messages"][-1].content}


def pdf_text_blob(pdf: bytes) -> str:
    """All page-content text: inflate every stream, keep the text-showing operators."""
    chunks = []
    for raw in re.findall(rb"stream\r?\n(.*?)\r?\nendstream", pdf, re.S):
        try:
            chunks.append(zlib.decompress(raw))
        except zlib.error:
            chunks.append(raw)
    blob = b"\n".join(chunks).decode("latin-1")
    return blob.replace("\\(", "(").replace("\\)", ")").replace("\\\\", "\\")  # PDF string escapes


def block(prompt: str, name: str) -> str:
    match = re.search(rf"===== {name} ANSWER =====\n(.*?)\n===== END {name} ANSWER =====", prompt, re.S)
    assert match, prompt
    return match.group(1)


class _FakeStructured:
    def __init__(self, judge: "FakeJudge"):
        self._judge = judge

    async def ainvoke(self, messages):
        return self._judge.decide(messages)


class FakeJudge:
    """Prefers the answer containing GOOD; flags the answer containing WRONG."""

    def __init__(self, decide=None):
        self.prompts: list[str] = []
        self.schemas: list[type] = []
        self._override = decide

    def with_structured_output(self, schema):
        self.schemas.append(schema)
        return _FakeStructured(self)

    def decide(self, messages):
        text = "\n".join(m.content for m in messages)
        self.prompts.append(text)
        if self._override is not None:
            return self._override(text)
        first, second = block(text, "FIRST"), block(text, "SECOND")
        if "GOOD" in first and "GOOD" not in second:
            winner = "first"
        elif "GOOD" in second and "GOOD" not in first:
            winner = "second"
        else:
            winner = "tie"
        return JudgeVerdict(
            winner=winner,
            first_has_unique_factual_error="WRONG" in first,
            second_has_unique_factual_error="WRONG" in second,
            rationale=f"picked {winner}",
        )
