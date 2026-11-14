"""AC1: `run` records each tool call's input and output as judge evidence, bounded and flagged.

Agents are real LangGraph ReAct graphs on a scripted chat model, so the tool spans are genuine
OpenInference spans captured by the real `capture_spans`. No network, no real LLM.
"""

import json

import pytest
from langchain_core.tools import tool

from api.tracing import capture_spans, get_tracer
from evals.benchmark.architectures import ARCHITECTURES
from evals.benchmark.common import resolve_model
from evals.benchmark.metrics import (
    EVIDENCE_INPUT_CAP,
    EVIDENCE_OUTPUT_CAP,
    EVIDENCE_TOTAL_CAP,
    SPAN_ATTRIBUTE_CAP,
    tool_output_text,
)
from evals.benchmark.questions import Question
from evals.benchmark.runner import run_benchmark
from evals.benchmark.runs import load_run
from evals.benchmark.schemas import QuestionResult, ToolEvidence
from tests.benchmark.helpers import FakeAgent, ToolAgent, write_run

pytestmark = pytest.mark.eval_unit

SECRET = "sk-test-secret-value-1234567890"


@tool
async def get_filing(topic: str) -> str:
    """Fetch a filing section."""
    return f"| Revenue | $94.9B | {topic} |"


@tool
async def tavily_search(query: str) -> str:
    """Inner search."""
    return f"inner result for {query}"


@tool
async def web_search(query: str) -> str:
    """Outer search that delegates to the inner tool (nested TOOL spans)."""
    return "OUTER: " + await tavily_search.ainvoke({"query": query})


@tool
async def dump(size: int) -> str:
    """Return `size` characters (a multi-line table stand-in)."""
    return ("0123456789\n" * (size // 11 + 1))[:size]


@tool
async def leak(topic: str) -> str:
    """Returns text containing a configured secret."""
    return f"debug: token={SECRET} for {topic}"


@tool
async def broken(topic: str) -> str:
    """Always fails."""
    raise ValueError(f"upstream down for {topic}")


ALL_TOOLS = [get_filing, web_search, tavily_search, dump, leak, broken]


@pytest.fixture
def model():
    return resolve_model(None)


@pytest.fixture(autouse=True)
def _clean_registry(monkeypatch):
    monkeypatch.setenv("TAVILY_API_KEY", SECRET)
    before = dict(ARCHITECTURES)
    yield
    ARCHITECTURES.clear()
    ARCHITECTURES.update(before)


async def _run(model, tmp_path, calls, qid="q1"):
    ARCHITECTURES["tools"] = lambda ticker, model_id: _async(ToolAgent(ALL_TOOLS, calls))
    out = tmp_path / "run"
    await run_benchmark(
        "tools", model, [Question(id=qid, ticker="AAPL", category="technicals", question="Q?")], out,
        progress=lambda _m: None,
    )
    return out, load_run(out).questions[qid]


async def _async(value):
    return value


class TestEvidenceRecorded:
    async def test_each_tool_call_input_and_output_is_stored_from_the_spans(self, model, tmp_path):
        ARCHITECTURES["fake"] = lambda ticker, model_id: _async(FakeAgent("Answer."))
        out = tmp_path / "run"
        await run_benchmark(
            "fake", model, [Question(id="q1", ticker="AAPL", category="technicals", question="Q?")], out,
            progress=lambda _m: None,
        )
        result = load_run(out).questions["q1"]
        assert [(e.tool, e.output, e.error, e.truncated) for e in result.evidence] == [
            ("lookup_fact", "fact about x", False, False)
        ]
        assert "x" in result.evidence[0].input
        assert result.evidence_truncated is False
        # Plain text, not the serialized ToolMessage the span carries.
        on_disk = json.loads((out / "questions" / "q1.json").read_text())
        assert on_disk["evidence"][0]["output"] == "fact about x"

    async def test_calls_are_stored_in_order(self, model, tmp_path):
        _, result = await _run(
            model, tmp_path,
            [("get_filing", {"topic": "FY24"}), ("get_filing", {"topic": "FY23"})],
        )
        assert [t.name for t in result.tool_calls] == ["get_filing", "get_filing"]
        assert [(e.tool, e.output) for e in result.evidence] == [
            ("get_filing", "| Revenue | $94.9B | FY24 |"),
            ("get_filing", "| Revenue | $94.9B | FY23 |"),
        ]

    async def test_nested_tool_spans_keep_only_the_outermost_output(self, model, tmp_path):
        _, result = await _run(model, tmp_path, [("web_search", {"query": "apple news"})])
        assert [t.name for t in result.tool_calls] == ["web_search", "tavily_search"]
        assert [(e.tool, e.output) for e in result.evidence] == [
            ("web_search", "OUTER: inner result for apple news")
        ]

    async def test_a_failing_tool_is_recorded_with_its_error(self, model, tmp_path):
        _, result = await _run(model, tmp_path, [("broken", {"topic": "AAPL"})])
        (item,) = result.evidence
        assert item.tool == "broken" and item.error is True
        assert "upstream down for AAPL" in item.output

    async def test_secrets_never_reach_the_evidence_files(self, model, tmp_path):
        out, result = await _run(model, tmp_path, [("leak", {"topic": "x"})])
        assert SECRET not in "".join(p.read_text() for p in out.rglob("*.json"))
        assert "[redacted]" in result.evidence[0].output


class TestEvidenceBounds:
    async def test_outputs_above_the_apps_8192_char_span_cap_are_kept_whole_up_to_the_evidence_cap(
        self, model, tmp_path
    ):
        _, result = await _run(model, tmp_path, [("dump", {"size": 20_000})])
        (item,) = result.evidence
        assert len(item.output) == 20_000 and item.truncated is False
        assert item.output.startswith("0123456789\n0123456789\n")  # unwrapped text, no JSON escaping
        assert result.evidence_truncated is False

    async def test_an_output_beyond_the_evidence_cap_is_cut_and_flagged(self, model, tmp_path):
        _, result = await _run(model, tmp_path, [("dump", {"size": SPAN_ATTRIBUTE_CAP + 10_000})])
        (item,) = result.evidence
        assert len(item.output) == EVIDENCE_OUTPUT_CAP
        assert item.output == ("0123456789\n" * 3000)[:EVIDENCE_OUTPUT_CAP]  # a clean prefix of the real output
        assert item.truncated is True and result.evidence_truncated is True

    async def test_the_total_per_question_is_capped_and_later_calls_are_still_listed(self, model, tmp_path):
        calls = [("dump", {"size": EVIDENCE_OUTPUT_CAP}) for _ in range(5)]
        _, result = await _run(model, tmp_path, calls)
        assert len(result.evidence) == 5
        assert sum(len(e.input) + len(e.output) for e in result.evidence) <= EVIDENCE_TOTAL_CAP
        assert [e.truncated for e in result.evidence] == [False, False, False, True, True]
        assert result.evidence[-1].output == ""
        assert result.evidence_truncated is True

    async def test_a_huge_tool_input_is_cut(self, model, tmp_path):
        _, result = await _run(model, tmp_path, [("get_filing", {"topic": "T" * 5_000})])
        (item,) = result.evidence
        assert len(item.input) == EVIDENCE_INPUT_CAP and item.truncated is True


class TestCaptureSpansCap:
    def _attribute_length(self, **kwargs) -> int:
        with capture_spans(**kwargs) as captured:
            with get_tracer().start_as_current_span("s", attributes={"text": "x" * 100_000}):
                pass
        return len(captured.spans[0].attributes["text"])

    def test_the_default_cap_is_unchanged_and_the_keyword_raises_it_for_that_capture_only(self):
        assert self._attribute_length() == 8192
        assert self._attribute_length(max_attribute_length=SPAN_ATTRIBUTE_CAP) == SPAN_ATTRIBUTE_CAP
        assert self._attribute_length() == 8192


class TestToolOutputText:
    def _wrap(self, content: str) -> str:
        return json.dumps({"type": "tool", "data": {"content": content, "name": "t", "status": "success"}})

    def test_unwraps_the_serialized_tool_message(self):
        assert tool_output_text(self._wrap("line1\nline2 \"q\"")) == ("line1\nline2 \"q\"", False)

    def test_recovers_the_content_written_before_the_span_cap_cut_the_json(self):
        wrapped = self._wrap("a\nb \"c\" " * 100)
        cut = wrapped[:300]
        text, truncated = tool_output_text(cut)
        assert truncated is True
        assert ("a\nb \"c\" " * 100).startswith(text) and len(text) > 150

    def test_a_cut_in_the_middle_of_an_escape_is_dropped_cleanly(self):
        wrapped = self._wrap("é\n" * 50)
        for end in range(60, 80):
            text, _ = tool_output_text(wrapped[:end])
            assert ("é\n" * 50).startswith(text)

    def test_content_blocks_are_joined_and_unknown_shapes_fall_back_to_the_raw_text(self):
        blocks = json.dumps({"type": "tool", "data": {"content": [{"type": "text", "text": "one"}, {"type": "text", "text": "two"}]}})
        assert tool_output_text(blocks)[0] == "one\ntwo"
        assert tool_output_text("plain text output") == ("plain text output", False)


class TestOlderQuestionFilesStillLoad:
    def test_a_question_json_without_the_evidence_fields_loads_with_evidence_unrecorded(self, tmp_path):
        run_dir = write_run(tmp_path, "old")
        path = run_dir / "questions" / "q-risk.json"
        raw = json.loads(path.read_text())
        raw.pop("evidence", None)
        raw.pop("evidence_truncated", None)
        path.write_text(json.dumps(raw))

        result = load_run(run_dir).questions["q-risk"]
        assert result.evidence is None  # "not recorded", distinct from "no tool calls"
        assert result.evidence_truncated is False

    def test_no_tool_calls_is_an_empty_list_not_none(self):
        assert QuestionResult.model_validate(
            {"id": "a", "ticker": "AAPL", "category": "c", "question": "q", "status": "ok", "evidence": []}
        ).evidence == []
        assert ToolEvidence(tool="t").output == ""
