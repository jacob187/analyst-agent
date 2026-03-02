"""LLM-judge evaluations for three new SEC tools.

Marker: eval_tools — requires SEC_HEADER (EDGAR) and GOOGLE_API_KEY (judge LLM).

Design
------
Each tool is evaluated at two levels:

1. Structural (keyword-based, no LLM)
   Fast checks that the raw text contains domain-appropriate terms.
   These catch wiring bugs (wrong section retrieved, error message returned).

2. LLM judge
   An LLM reads the tool output and answers a fixed set of yes/no questions
   via a Pydantic-parsed response.  The judge model is the session-scoped
   `llm` fixture so all three evaluations share one LLM instance.

   Why LLM judgment rather than keyword checks alone:
   - Section content varies by company — AAPL may phrase things differently
     from MSFT or GOOGL.
   - We want to verify *semantic* correctness, not just string presence.
   - Judge failures produce a human-readable `verdict` field that makes
     debugging easy.

Session-scoped output fixtures
-------------------------------
Each tool is invoked once per test session via a session-scoped fixture that
depends on `tools_dict` (which depends on `sec_header`).  If either API key
is missing, the fixture auto-skips and every dependent test is skipped.
The edgartools in-process cache ensures the EDGAR filing is only fetched once
even if multiple test classes use the same underlying data.
"""

import pytest
from pydantic import BaseModel, Field
from langchain_core.output_parsers import PydanticOutputParser
from langchain_core.prompts import ChatPromptTemplate


# ── LLM judge model ──────────────────────────────────────────────────────────

class SectionEval(BaseModel):
    is_correct_section: bool = Field(
        description="True if the content matches the expected SEC filing section type"
    )
    has_meaningful_content: bool = Field(
        description="True if the content contains real, substantive information beyond headers or boilerplate"
    )
    verdict: str = Field(
        description="One sentence explaining the evaluation"
    )


# ── Judge helper ─────────────────────────────────────────────────────────────

_JUDGE_SYSTEM = (
    "You are a strict financial analyst evaluating SEC filing content. "
    "Respond ONLY with the requested JSON object — no prose, no markdown fences."
)

_JUDGE_USER = """You are evaluating content extracted from a SEC filing.

Expected section: {section}

Content (first 2500 characters):
{content}

Evaluation criteria:
- is_correct_section: Does this content describe what the expected section should describe?
  Mark False if it clearly belongs to a different section or is an error message.
- has_meaningful_content: Does it go beyond section headers and boilerplate?
  Mark False for empty results, pure "not applicable" statements, or error strings.

{format_instructions}"""


def _llm_judge(llm, content: str, section: str) -> SectionEval:
    parser = PydanticOutputParser(pydantic_object=SectionEval)
    prompt = ChatPromptTemplate.from_messages([
        ("system", _JUDGE_SYSTEM),
        ("user", _JUDGE_USER),
    ]).partial(
        section=section,
        content=content[:2500],
        format_instructions=parser.get_format_instructions(),
    )
    return (prompt | llm | parser).invoke({})


# ── Session-scoped tool output fixtures ─────────────────────────────────────

@pytest.fixture(scope="session")
def business_text(tools_dict):
    return tools_dict["get_business_overview"]("")


@pytest.fixture(scope="session")
def cybersecurity_text(tools_dict):
    return tools_dict["get_cybersecurity_disclosure"]("")


@pytest.fixture(scope="session")
def legal_text(tools_dict):
    return tools_dict["get_legal_proceedings"]("")


# ── get_business_overview ────────────────────────────────────────────────────

@pytest.mark.eval_tools
class TestBusinessOverview:
    """10-K Item 1 — Business: products, services, segments, markets."""

    def test_returns_substantial_text(self, business_text):
        assert isinstance(business_text, str)
        assert len(business_text) > 200, f"Too short: {business_text[:100]}"

    def test_no_error_string(self, business_text):
        assert "Failed to" not in business_text
        assert "not found" not in business_text.lower()

    def test_contains_business_vocabulary(self, business_text):
        """Item 1 must contain at least two domain-appropriate terms."""
        terms = ["product", "service", "market", "revenue", "customer",
                 "segment", "operations", "business"]
        matches = [t for t in terms if t in business_text.lower()]
        assert len(matches) >= 2, f"Too few business terms ({matches}) in: {business_text[:200]}"

    def test_llm_judges_correct_section(self, business_text, llm):
        result = _llm_judge(
            llm, business_text,
            "10-K Item 1 — Business: overview of the company's products, services, "
            "business segments, and operating environment"
        )
        assert result.is_correct_section, (
            f"LLM: wrong section. Verdict: {result.verdict}"
        )

    def test_llm_judges_meaningful_content(self, business_text, llm):
        result = _llm_judge(
            llm, business_text,
            "10-K Item 1 — Business: overview of the company's products, services, "
            "business segments, and operating environment"
        )
        assert result.has_meaningful_content, (
            f"LLM: not substantive. Verdict: {result.verdict}"
        )


# ── get_cybersecurity_disclosure ─────────────────────────────────────────────

@pytest.mark.eval_tools
class TestCybersecurityDisclosure:
    """10-K Item 1C — Cybersecurity: risk management, governance, oversight."""

    def test_returns_nonempty_text(self, cybersecurity_text):
        assert isinstance(cybersecurity_text, str)
        assert len(cybersecurity_text) > 50, f"Too short: {cybersecurity_text[:100]}"

    def test_no_error_string(self, cybersecurity_text):
        assert "Failed to" not in cybersecurity_text

    def test_contains_security_vocabulary(self, cybersecurity_text):
        """Item 1C must mention security governance or risk management."""
        terms = ["cybersecurity", "security", "risk", "threat", "incident",
                 "information security", "governance", "oversight"]
        lower = cybersecurity_text.lower()
        matches = [t for t in terms if t in lower]
        assert len(matches) >= 2, (
            f"Too few security terms ({matches}) in: {cybersecurity_text[:200]}"
        )

    def test_llm_judges_correct_section(self, cybersecurity_text, llm):
        result = _llm_judge(
            llm, cybersecurity_text,
            "10-K Item 1C — Cybersecurity: the company's cybersecurity risk management "
            "strategy, governance structure, and oversight processes"
        )
        assert result.is_correct_section, (
            f"LLM: wrong section. Verdict: {result.verdict}"
        )

    def test_llm_judges_meaningful_content(self, cybersecurity_text, llm):
        result = _llm_judge(
            llm, cybersecurity_text,
            "10-K Item 1C — Cybersecurity: the company's cybersecurity risk management "
            "strategy, governance structure, and oversight processes"
        )
        assert result.has_meaningful_content, (
            f"LLM: not substantive. Verdict: {result.verdict}"
        )


# ── get_legal_proceedings ────────────────────────────────────────────────────

@pytest.mark.eval_tools
class TestLegalProceedings:
    """10-K Item 3 — Legal Proceedings: litigation, regulatory actions, investigations."""

    def test_returns_substantial_text(self, legal_text):
        assert isinstance(legal_text, str)
        assert len(legal_text) > 100, f"Too short: {legal_text[:100]}"

    def test_no_error_string(self, legal_text):
        assert "Failed to" not in legal_text

    def test_contains_legal_vocabulary(self, legal_text):
        """Item 3 must contain at least two legal domain terms."""
        terms = ["proceeding", "litigation", "legal", "investigation",
                 "regulatory", "claim", "lawsuit", "court", "action", "compliance"]
        lower = legal_text.lower()
        matches = [t for t in terms if t in lower]
        assert len(matches) >= 2, (
            f"Too few legal terms ({matches}) in: {legal_text[:200]}"
        )

    def test_llm_judges_correct_section(self, legal_text, llm):
        result = _llm_judge(
            llm, legal_text,
            "10-K Item 3 — Legal Proceedings: significant pending lawsuits, "
            "regulatory investigations, and government enforcement actions"
        )
        assert result.is_correct_section, (
            f"LLM: wrong section. Verdict: {result.verdict}"
        )

    def test_llm_judges_meaningful_content(self, legal_text, llm):
        result = _llm_judge(
            llm, legal_text,
            "10-K Item 3 — Legal Proceedings: significant pending lawsuits, "
            "regulatory investigations, and government enforcement actions"
        )
        assert result.has_meaningful_content, (
            f"LLM: not substantive. Verdict: {result.verdict}"
        )
