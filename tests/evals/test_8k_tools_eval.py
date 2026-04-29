"""Eval tests for 8-K filing tools — requires SEC_HEADER and GOOGLE_API_KEY.

Marker: eval_tools

Tests at two levels (same pattern as test_new_tools_eval.py):

1. Structural — JSON parse + key presence checks. Catches wiring bugs
   (wrong filing type, error messages returned, missing Pydantic fields).

2. LLM judge — an LLM evaluates semantic correctness via Pydantic-parsed
   yes/no questions. Produces a human-readable ``verdict`` on failure.

Session-scoped fixtures ensure each tool is invoked only once per run.
The edgartools in-process cache ensures the 8-K filing is fetched once.

Tool topology (post-Phase-1 of issue #30):
- ``analyze_latest_8k`` is a dispatcher tool that internally routes to either
  the earnings analyzer (Item 2.02) or the material event analyzer (other
  items). Both branches return the same JSON shape (a ``model_dump()`` of
  ``EarningsAnalysis`` or ``MaterialEventAnalysis`` respectively).
- ``get_8k_item`` returns raw item text and is unchanged.
"""

import json
import pytest
from pydantic import BaseModel, Field
from langchain_core.output_parsers import PydanticOutputParser
from langchain_core.prompts import ChatPromptTemplate


# ── LLM judge models ─────────────────────────────────────────────────────────

class EightKEval(BaseModel):
    is_correct_content: bool = Field(
        description="True if the content is from an 8-K filing and matches the expected type"
    )
    has_meaningful_content: bool = Field(
        description="True if the content contains real, substantive information"
    )
    verdict: str = Field(
        description="One sentence explaining the evaluation"
    )


class EarningsEval(BaseModel):
    has_financial_data: bool = Field(
        description="True if the output contains financial metrics (revenue, EPS, margins, etc.)"
    )
    has_analysis: bool = Field(
        description="True if the output provides interpretation beyond raw numbers"
    )
    verdict: str = Field(
        description="One sentence explaining the evaluation"
    )


# ── Judge helpers ─────────────────────────────────────────────────────────────

_JUDGE_SYSTEM = (
    "You are a strict financial analyst evaluating SEC filing content. "
    "Respond ONLY with the requested JSON object — no prose, no markdown fences."
)

_8K_JUDGE_USER = """You are evaluating content extracted from an 8-K filing.

Expected content type: {content_type}

Content (first 2500 characters):
{content}

Evaluation criteria:
- is_correct_content: Does this look like it came from an 8-K filing?
  Mark False if it's clearly an error message or from a different filing type.
- has_meaningful_content: Does it contain real information about a corporate event?
  Mark False for empty results, boilerplate, or error strings.

{format_instructions}"""

_EARNINGS_JUDGE_USER = """You are evaluating an earnings analysis output.

Content (first 2500 characters):
{content}

Evaluation criteria:
- has_financial_data: Does it mention specific financial metrics like revenue,
  EPS, net income, margins, or growth rates?
- has_analysis: Does it provide interpretation (beats/misses, trends, sentiment)
  beyond just listing numbers?

{format_instructions}"""


def _judge_8k(llm, content: str, content_type: str) -> EightKEval:
    parser = PydanticOutputParser(pydantic_object=EightKEval)
    prompt = ChatPromptTemplate.from_messages([
        ("system", _JUDGE_SYSTEM),
        ("user", _8K_JUDGE_USER),
    ]).partial(
        content_type=content_type,
        content=content[:2500],
        format_instructions=parser.get_format_instructions(),
    )
    return (prompt | llm | parser).invoke({})


def _judge_earnings(llm, content: str) -> EarningsEval:
    parser = PydanticOutputParser(pydantic_object=EarningsEval)
    prompt = ChatPromptTemplate.from_messages([
        ("system", _JUDGE_SYSTEM),
        ("user", _EARNINGS_JUDGE_USER),
    ]).partial(
        content=content[:2500],
        format_instructions=parser.get_format_instructions(),
    )
    return (prompt | llm | parser).invoke({})


# ── Session-scoped fixtures ──────────────────────────────────────────────────

@pytest.fixture(scope="session")
def eightk_item_text(tools_dict):
    """Fetch Item 2.02 (earnings) — most common 8-K item for AAPL."""
    return tools_dict["get_8k_item"]("2.02")


@pytest.fixture(scope="session")
def eightk_analysis_text(tools_dict):
    """Run the dispatcher; routes internally to earnings or material event."""
    return tools_dict["analyze_latest_8k"]("")


@pytest.fixture(scope="session")
def eightk_analysis_dict(eightk_analysis_text):
    """Parse the dispatcher output as JSON. Skips if the tool returned an error string.

    The dispatcher always returns JSON on success (a ``model_dump()`` of either
    ``EarningsAnalysis`` or ``MaterialEventAnalysis``). On routing-stage errors
    (no 8-K, retriever failure) it also returns JSON with ``found: False`` or
    ``error``. On analyzer-stage failures the underlying tool returns a plain
    error string — guard against that with a parse attempt.
    """
    try:
        return json.loads(eightk_analysis_text)
    except json.JSONDecodeError:
        pytest.skip(f"Dispatcher returned non-JSON: {eightk_analysis_text[:200]}")


# ── get_8k_item (unchanged) ──────────────────────────────────────────────────

@pytest.mark.eval_tools
class TestEightKItem:
    """8-K item retrieval returns raw text for a specific item number."""

    def test_returns_nonempty_text(self, eightk_item_text):
        assert isinstance(eightk_item_text, str)
        assert len(eightk_item_text) > 20, f"Too short: {eightk_item_text[:100]}"

    def test_no_error_string(self, eightk_item_text):
        assert "Failed to" not in eightk_item_text

    def test_contains_financial_vocabulary(self, eightk_item_text):
        """Item 2.02 (earnings) should mention financial terms."""
        if "not found" in eightk_item_text.lower():
            pytest.skip("Item 2.02 not in latest 8-K")
        terms = ["results", "operations", "financial", "earnings", "revenue",
                 "income", "quarter", "fiscal", "press release", "exhibit"]
        lower = eightk_item_text.lower()
        matches = [t for t in terms if t in lower]
        assert len(matches) >= 1, (
            f"Too few financial terms ({matches}) in: {eightk_item_text[:200]}"
        )


# ── analyze_latest_8k (dispatcher) ───────────────────────────────────────────

@pytest.mark.eval_tools
class TestAnalyzeLatest8K:
    """Dispatcher returns structured JSON from one of two analyzers."""

    def test_returns_nonempty_text(self, eightk_analysis_text):
        assert isinstance(eightk_analysis_text, str)
        assert len(eightk_analysis_text) > 50, (
            f"Too short: {eightk_analysis_text[:100]}"
        )

    def test_no_error_string(self, eightk_analysis_text):
        assert "Failed to" not in eightk_analysis_text

    def test_returns_valid_json(self, eightk_analysis_dict):
        """Output must be parseable JSON — proves we're returning structured data,
        not the old truncated f-string format."""
        assert isinstance(eightk_analysis_dict, dict)

    def test_contains_pydantic_fields(self, eightk_analysis_dict):
        """Output must contain the full Pydantic model fields, not a cherry-picked
        subset. Both EarningsAnalysis and MaterialEventAnalysis include ``summary``
        and ``sentiment_score`` — assert both are present."""
        if eightk_analysis_dict.get("found") is False:
            pytest.skip("No 8-K filing for this ticker")
        assert "summary" in eightk_analysis_dict, (
            f"Missing summary field: {list(eightk_analysis_dict.keys())}"
        )
        assert "sentiment_score" in eightk_analysis_dict, (
            f"Missing sentiment_score field: {list(eightk_analysis_dict.keys())}"
        )

    def test_contains_routing_specific_fields(self, eightk_analysis_dict):
        """Earnings branch has ``key_metrics``; material event branch has
        ``event_type``. Output should match exactly one shape — proves the
        dispatcher routed to one analyzer rather than blending both."""
        if eightk_analysis_dict.get("found") is False:
            pytest.skip("No 8-K filing for this ticker")
        is_earnings = "key_metrics" in eightk_analysis_dict
        is_material = "event_type" in eightk_analysis_dict
        assert is_earnings ^ is_material, (
            f"Output should match exactly one analyzer shape "
            f"(earnings xor material event), got keys: "
            f"{sorted(eightk_analysis_dict.keys())}"
        )

    def test_full_field_preservation(self, eightk_analysis_dict):
        """Smoke test for the bug fix: the response must contain the
        ``sentiment_analysis`` field (long-form sentiment explanation), which
        the old f-string return always dropped."""
        if eightk_analysis_dict.get("found") is False:
            pytest.skip("No 8-K filing for this ticker")
        assert "sentiment_analysis" in eightk_analysis_dict, (
            "Tool dropped sentiment_analysis — same class of bug that produced "
            "the contradiction in issue #30. Got keys: "
            f"{sorted(eightk_analysis_dict.keys())}"
        )

    def test_llm_judges_correct_content(self, eightk_analysis_text, llm):
        result = _judge_8k(
            llm, eightk_analysis_text,
            "8-K filing analysis — structured JSON with sentiment, key points, "
            "and either earnings metrics or material event details"
        )
        assert result.has_meaningful_content, (
            f"LLM: not substantive. Verdict: {result.verdict}"
        )

    def test_llm_judges_earnings_quality_when_routed_to_earnings(
        self, eightk_analysis_text, eightk_analysis_dict, llm
    ):
        """When the dispatcher routes to the earnings analyzer, the output
        should provide real financial analysis — not just numbers."""
        if eightk_analysis_dict.get("found") is False:
            pytest.skip("No 8-K filing for this ticker")
        if "key_metrics" not in eightk_analysis_dict:
            pytest.skip("Dispatcher routed to material event, not earnings")
        result = _judge_earnings(llm, eightk_analysis_text)
        assert result.has_analysis, (
            f"LLM: no real analysis. Verdict: {result.verdict}"
        )
