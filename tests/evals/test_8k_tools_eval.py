"""Eval tests for 8-K filing tools — requires SEC_HEADER and GOOGLE_API_KEY.

Marker: eval_tools

Tests at two levels (same pattern as test_new_tools_eval.py):

1. Structural — keyword checks that the raw output contains expected terms.
   Catches wiring bugs (wrong filing type, error messages returned).

2. LLM judge — an LLM evaluates semantic correctness via Pydantic-parsed
   yes/no questions.  Produces a human-readable ``verdict`` on failure.

Session-scoped fixtures ensure each tool is invoked only once per run.
The edgartools in-process cache ensures the 8-K filing is fetched once.
"""

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
def eightk_overview_text(tools_dict):
    return tools_dict["get_8k_overview"]("")


@pytest.fixture(scope="session")
def eightk_item_text(tools_dict):
    """Fetch Item 2.02 (earnings) — most common 8-K item for AAPL."""
    return tools_dict["get_8k_item"]("2.02")


@pytest.fixture(scope="session")
def earnings_summary_text(tools_dict):
    return tools_dict["get_earnings_summary"]("")


@pytest.fixture(scope="session")
def material_event_text(tools_dict):
    return tools_dict["get_material_event_summary"]("")


# ── get_8k_overview ──────────────────────────────────────────────────────────

@pytest.mark.eval_tools
class TestEightKOverview:
    """8-K overview tool returns filing metadata and event classification."""

    def test_returns_nonempty_text(self, eightk_overview_text):
        assert isinstance(eightk_overview_text, str)
        assert len(eightk_overview_text) > 50, f"Too short: {eightk_overview_text[:100]}"

    def test_no_error_string(self, eightk_overview_text):
        assert "Failed to" not in eightk_overview_text
        assert "No 8-K" not in eightk_overview_text

    def test_contains_event_type(self, eightk_overview_text):
        """Overview must mention an event type classification."""
        assert "Event Type:" in eightk_overview_text

    def test_contains_items(self, eightk_overview_text):
        """Overview must list 8-K item numbers."""
        assert "Items:" in eightk_overview_text
        # At least one item number in X.XX format
        import re
        assert re.search(r"\d\.\d{2}", eightk_overview_text), (
            f"No item number (X.XX) found in: {eightk_overview_text[:200]}"
        )

    def test_llm_judges_correct_content(self, eightk_overview_text, llm):
        result = _judge_8k(
            llm, eightk_overview_text,
            "8-K filing overview — event classification, item numbers, and filing dates"
        )
        assert result.is_correct_content, (
            f"LLM: not 8-K content. Verdict: {result.verdict}"
        )


# ── get_8k_item ──────────────────────────────────────────────────────────────

@pytest.mark.eval_tools
class TestEightKItem:
    """8-K item retrieval returns raw text for a specific item number."""

    def test_returns_nonempty_text(self, eightk_item_text):
        assert isinstance(eightk_item_text, str)
        # Item 2.02 might not always be present, but for AAPL it should be
        assert len(eightk_item_text) > 20, f"Too short: {eightk_item_text[:100]}"

    def test_no_error_string(self, eightk_item_text):
        assert "Failed to" not in eightk_item_text

    def test_contains_financial_vocabulary(self, eightk_item_text):
        """Item 2.02 (earnings) should mention financial terms."""
        # If the item wasn't found, skip this check
        if "not found" in eightk_item_text.lower():
            pytest.skip("Item 2.02 not in latest 8-K")
        terms = ["results", "operations", "financial", "earnings", "revenue",
                 "income", "quarter", "fiscal", "press release", "exhibit"]
        lower = eightk_item_text.lower()
        matches = [t for t in terms if t in lower]
        assert len(matches) >= 1, (
            f"Too few financial terms ({matches}) in: {eightk_item_text[:200]}"
        )


# ── get_earnings_summary ─────────────────────────────────────────────────────

@pytest.mark.eval_tools
class TestEarningsSummary:
    """Earnings summary tool returns structured LLM analysis of 8-K Item 2.02."""

    def test_returns_nonempty_text(self, earnings_summary_text):
        assert isinstance(earnings_summary_text, str)
        assert len(earnings_summary_text) > 50, f"Too short: {earnings_summary_text[:100]}"

    def test_no_error_string(self, earnings_summary_text):
        assert "Failed to" not in earnings_summary_text

    def test_contains_analysis_structure(self, earnings_summary_text):
        """Output should have structured sections."""
        # If no earnings data, it's fine — just check it's a clear message
        if "No earnings data" in earnings_summary_text:
            assert "Reason:" in earnings_summary_text or "reason" in earnings_summary_text.lower()
            return
        # Otherwise expect structured analysis
        assert "Earnings Analysis:" in earnings_summary_text or "Sentiment:" in earnings_summary_text

    def test_contains_metrics_or_clear_message(self, earnings_summary_text):
        """Must contain financial metrics or a clear 'no earnings' message."""
        if "No earnings data" in earnings_summary_text:
            return  # acceptable
        terms = ["revenue", "eps", "income", "margin", "growth", "beat", "miss",
                 "metric", "billion", "million", "%"]
        lower = earnings_summary_text.lower()
        matches = [t for t in terms if t in lower]
        assert len(matches) >= 2, (
            f"Too few financial terms ({matches}) in: {earnings_summary_text[:300]}"
        )

    def test_llm_judges_analysis_quality(self, earnings_summary_text, llm):
        """LLM judges whether the earnings analysis is substantive."""
        if "No earnings data" in earnings_summary_text:
            pytest.skip("No earnings in latest 8-K — cannot judge analysis quality")
        result = _judge_earnings(llm, earnings_summary_text)
        assert result.has_analysis, (
            f"LLM: no real analysis. Verdict: {result.verdict}"
        )


# ── get_material_event_summary ───────────────────────────────────────────────

@pytest.mark.eval_tools
class TestMaterialEventSummary:
    """Material event summary returns structured analysis of non-earnings 8-K."""

    def test_returns_nonempty_text(self, material_event_text):
        assert isinstance(material_event_text, str)
        assert len(material_event_text) > 50, f"Too short: {material_event_text[:100]}"

    def test_no_error_string(self, material_event_text):
        assert "Failed to" not in material_event_text

    def test_contains_event_classification(self, material_event_text):
        """Output should classify the event type."""
        assert "Material Event" in material_event_text or "Sentiment:" in material_event_text

    def test_contains_impact_assessment(self, material_event_text):
        """Output should assess impact on the company."""
        assert "Impact:" in material_event_text or "impact" in material_event_text.lower()

    def test_llm_judges_correct_content(self, material_event_text, llm):
        result = _judge_8k(
            llm, material_event_text,
            "8-K material event analysis — classification, key points, and impact assessment"
        )
        assert result.has_meaningful_content, (
            f"LLM: not substantive. Verdict: {result.verdict}"
        )
