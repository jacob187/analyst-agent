"""Unit tests for SEC tool return shape — no API keys required.

Marker: eval_unit

The bug in issue #30 had two compounding causes:

1. The planner picked two overlapping 8-K tools (rule-based overview +
   LLM-based material event analyzer) that classified the same filing
   differently — fixed by collapsing 8-K tools into a single dispatcher
   (``analyze_latest_8k``).

2. Every LLM-analysis tool collapsed its Pydantic model into a hand-formatted
   ``f"..."`` string, cherry-picking 2-5 fields and silently dropping the
   rest (e.g. ``MDnAAnalysis.financial_highlights``,
   ``BalanceSheetAnalysis.liquidity_analysis``,
   ``EarningsAnalysis.sentiment_analysis``). By the time the synthesizer ran,
   the structured data was gone and reconciliation across tool outputs was
   impossible — fixed by returning ``json.dumps(model_dump_dict, indent=2)``.

These unit tests prove that fix (2) holds: each tool that wraps an LLM
analysis returns parseable JSON containing **every** field of its Pydantic
model. The tests pre-populate the in-process ``_processed_cache`` with a
hand-crafted ``model_dump()`` dict so the tool's return path runs without
any LLM call. This is a fake (a real dict in a real cache), not a mock of
project code — see CLAUDE.md testing philosophy.
"""

import json
import pytest

from agents.tools import sec_tools
from agents.sec_workflow.sec_llm_models import (
    EarningsAnalysis,
    MaterialEventAnalysis,
    MDnAAnalysis,
    RiskFactorAnalysis,
    BalanceSheetAnalysis,
)


TICKER = "TESTCO"


@pytest.fixture(autouse=True)
def reset_cache():
    """Each test gets an empty processed cache so seeded fixtures don't leak."""
    sec_tools._processed_cache.clear()
    yield
    sec_tools._processed_cache.clear()


def _seed(ticker: str, key: str, model: object) -> None:
    """Seed the in-process cache with a Pydantic model's ``model_dump()``."""
    sec_tools._processed_cache.setdefault(ticker, {})
    sec_tools._processed_cache[ticker][f"{ticker}_{key}"] = model.model_dump()


def _full_earnings_analysis() -> EarningsAnalysis:
    return EarningsAnalysis(
        summary="Q4 results beat expectations across the board.",
        key_metrics=[
            "Revenue $94.9B (+6%)",
            "Diluted EPS $2.40 (+16%)",
            "Operating Income $33.9B (+24%)",
            "Gross margin 46.6% (+220bps)",
            "Services revenue $26.3B (+12%)",
            "iPhone revenue $69.7B (+5%)",
            "Mac revenue $9.0B (+2%)",
            "Free cash flow $30.1B",
        ],
        beats_misses=[
            "Beat consensus EPS by $0.08",
            "Beat consensus revenue by $1.2B",
            "Services beat by $400M",
        ],
        guidance="Q1 revenue growth in mid-single digits.",
        sentiment_score=7.5,
        sentiment_analysis=(
            "Strong all-around quarter with margin expansion driven by services "
            "mix and operating leverage. Guidance suggests sustained momentum."
        ),
        filing_metadata={"filing_date": "2026-04-07", "period": "Q4 2025"},
    )


def _full_material_event_analysis() -> MaterialEventAnalysis:
    return MaterialEventAnalysis(
        summary="Executive compensation overhaul covering CFO, President, and two SVPs.",
        event_type="executive_compensation",
        key_points=[
            "CFO Anat Ashkenazi PSU grant valued at $34M",
            "President Ruth Porat GSU grant valued at $47.7M",
            "Vesting schedule: 4-year cliff with 25% annual",
            "Performance metrics tied to TSR vs S&P 500",
        ],
        impact_assessment=(
            "Aligns leadership pay with long-term shareholder returns; "
            "the magnitudes are notable but consistent with FAANG peers."
        ),
        sentiment_score=2.0,
        sentiment_analysis=(
            "Modestly positive — the structure is performance-based, but "
            "the magnitudes will draw shareholder scrutiny."
        ),
        filing_metadata={"filing_date": "2026-04-07", "item": "5.02"},
    )


def _full_mda_analysis() -> MDnAAnalysis:
    return MDnAAnalysis(
        summary="Management notes services growth and easing supply pressure.",
        key_points=[
            "Services growth accelerating",
            "Supply chain stabilizing",
            "AI capex materially elevated",
        ],
        financial_highlights=[
            "Operating margin expansion of 220bps",
            "Free cash flow $30.1B",
            "R&D spend $7.7B (+8%)",
            "Capex guide $14B for FY26",
        ],
        future_outlook="Mid-single-digit revenue growth, services-led mix shift.",
        sentiment_score=6.0,
        sentiment_analysis="Confident tone; concrete forward references.",
        form_type="10-Q",
        filing_metadata={"period": "Q4 2025"},
    )


def _full_risk_analysis() -> RiskFactorAnalysis:
    return RiskFactorAnalysis(
        summary="Risks are concentrated in supply chain, antitrust, and currency.",
        key_risks=[
            "Greater China revenue concentration",
            "DOJ antitrust litigation in services",
            "FX headwinds in EMEA",
            "Component pricing volatility",
        ],
        risk_categories={
            "regulatory": ["DOJ antitrust litigation", "EU DMA enforcement"],
            "operational": ["Supply chain concentration", "Component pricing"],
            "macro": ["FX headwinds", "Consumer demand softness"],
        },
        sentiment_score=-3.5,
        sentiment_analysis="Multiple material risks; antitrust is the highest-impact.",
        form_type="10-Q",
        filing_metadata={"period": "Q4 2025"},
    )


def _full_balance_sheet_analysis() -> BalanceSheetAnalysis:
    return BalanceSheetAnalysis(
        summary="Strong liquidity, growing buyback, manageable leverage.",
        key_metrics=[
            "Cash + marketable securities $162B",
            "Long-term debt $97B",
            "Shareholder equity $62B",
            "Total assets $352B",
        ],
        liquidity_analysis=(
            "Current ratio 1.05; quick ratio 0.98. Sufficient given working "
            "capital efficiency."
        ),
        solvency_analysis=(
            "Debt/equity 1.56; interest coverage >40x. Comfortable solvency."
        ),
        growth_trends="Cash from operations growing 4% YoY; capex flat.",
        financial_highlights=[
            "Buyback authorization $110B",
            "Dividend hike 4%",
            "Inventory days 18 (low for sector)",
        ],
        red_flags=[
            "Greater China receivables concentration",
        ],
    )


# ── Per-tool JSON shape tests ────────────────────────────────────────────────


@pytest.mark.eval_unit
class TestEarningsSummaryReturnsFullModel:
    """_tool_earnings_summary returns JSON containing every EarningsAnalysis field."""

    def test_returns_valid_json(self):
        _seed(TICKER, "earnings_summary", _full_earnings_analysis())
        out = sec_tools._tool_earnings_summary(TICKER, llm=None)
        parsed = json.loads(out)
        assert isinstance(parsed, dict)

    def test_preserves_all_pydantic_fields(self):
        model = _full_earnings_analysis()
        _seed(TICKER, "earnings_summary", model)
        parsed = json.loads(sec_tools._tool_earnings_summary(TICKER, llm=None))
        for field in EarningsAnalysis.model_fields:
            assert field in parsed, f"Tool dropped field {field!r}"

    def test_preserves_all_key_metrics(self):
        """The old f-string truncated key_metrics to 5 — must now keep all 8."""
        model = _full_earnings_analysis()
        _seed(TICKER, "earnings_summary", model)
        parsed = json.loads(sec_tools._tool_earnings_summary(TICKER, llm=None))
        assert parsed["key_metrics"] == model.key_metrics
        assert len(parsed["key_metrics"]) == 8

    def test_preserves_sentiment_analysis(self):
        """The old f-string dropped sentiment_analysis entirely — proves it returns now."""
        model = _full_earnings_analysis()
        _seed(TICKER, "earnings_summary", model)
        parsed = json.loads(sec_tools._tool_earnings_summary(TICKER, llm=None))
        assert parsed["sentiment_analysis"] == model.sentiment_analysis


@pytest.mark.eval_unit
class TestMaterialEventSummaryReturnsFullModel:
    """_tool_material_event_summary returns JSON containing every field."""

    def test_returns_valid_json(self):
        _seed(TICKER, "material_event_summary", _full_material_event_analysis())
        parsed = json.loads(sec_tools._tool_material_event_summary(TICKER, llm=None))
        assert isinstance(parsed, dict)

    def test_preserves_all_pydantic_fields(self):
        _seed(TICKER, "material_event_summary", _full_material_event_analysis())
        parsed = json.loads(sec_tools._tool_material_event_summary(TICKER, llm=None))
        for field in MaterialEventAnalysis.model_fields:
            assert field in parsed, f"Tool dropped field {field!r}"

    def test_preserves_event_type_and_filing_metadata(self):
        """These two fields are what makes cross-tool reconciliation possible —
        they identify which filing the analysis describes."""
        model = _full_material_event_analysis()
        _seed(TICKER, "material_event_summary", model)
        parsed = json.loads(sec_tools._tool_material_event_summary(TICKER, llm=None))
        assert parsed["event_type"] == "executive_compensation"
        assert parsed["filing_metadata"]["filing_date"] == "2026-04-07"


@pytest.mark.eval_unit
class TestMDASummaryReturnsFullModel:
    """_tool_mda_summary returns JSON containing every MDnAAnalysis field —
    crucially ``financial_highlights`` (where dollar amounts live) which the
    old f-string return dropped entirely."""

    def test_preserves_all_pydantic_fields(self):
        _seed(TICKER, "mda_summary", _full_mda_analysis())
        parsed = json.loads(sec_tools._tool_mda_summary(TICKER, llm=None))
        for field in MDnAAnalysis.model_fields:
            assert field in parsed, f"Tool dropped field {field!r}"

    def test_preserves_financial_highlights(self):
        """``financial_highlights`` is where Diluted EPS / Operating Income /
        capex numbers live for MD&A. The old f-string dropped the entire field."""
        model = _full_mda_analysis()
        _seed(TICKER, "mda_summary", model)
        parsed = json.loads(sec_tools._tool_mda_summary(TICKER, llm=None))
        assert parsed["financial_highlights"] == model.financial_highlights
        assert len(parsed["financial_highlights"]) == 4


@pytest.mark.eval_unit
class TestRiskSummaryReturnsFullModel:
    """_tool_risk_factors_summary returns JSON containing the categorized
    ``risk_categories`` dict — which the old f-string return dropped."""

    def test_preserves_all_pydantic_fields(self):
        _seed(TICKER, "risk_summary", _full_risk_analysis())
        parsed = json.loads(sec_tools._tool_risk_factors_summary(TICKER, llm=None))
        for field in RiskFactorAnalysis.model_fields:
            assert field in parsed, f"Tool dropped field {field!r}"

    def test_preserves_risk_categories(self):
        model = _full_risk_analysis()
        _seed(TICKER, "risk_summary", model)
        parsed = json.loads(sec_tools._tool_risk_factors_summary(TICKER, llm=None))
        assert parsed["risk_categories"] == model.risk_categories


@pytest.mark.eval_unit
class TestBalanceSheetSummaryReturnsFullModel:
    """_tool_balance_sheet_summary returns JSON containing the four substantive
    fields the old f-string dropped: ``liquidity_analysis``, ``solvency_analysis``,
    ``growth_trends``, ``financial_highlights``."""

    def test_preserves_all_pydantic_fields(self):
        _seed(TICKER, "balance_summary", _full_balance_sheet_analysis())
        parsed = json.loads(sec_tools._tool_balance_sheet_summary(TICKER, llm=None))
        for field in BalanceSheetAnalysis.model_fields:
            assert field in parsed, f"Tool dropped field {field!r}"

    def test_preserves_liquidity_solvency_growth(self):
        model = _full_balance_sheet_analysis()
        _seed(TICKER, "balance_summary", model)
        parsed = json.loads(sec_tools._tool_balance_sheet_summary(TICKER, llm=None))
        assert parsed["liquidity_analysis"] == model.liquidity_analysis
        assert parsed["solvency_analysis"] == model.solvency_analysis
        assert parsed["growth_trends"] == model.growth_trends


# ── Dispatcher routing tests ────────────────────────────────────────────────


class _StubRetriever:
    """Stand-in for ``SECDataRetrieval`` used to exercise dispatcher routing.

    A fake (real class with hand-built return values), not a mock — see
    CLAUDE.md testing philosophy. Only the methods the dispatcher calls are
    implemented; reaching for any other method would raise AttributeError,
    which is the correct failure mode for an under-specified fake.
    """

    def __init__(self, has_earnings: bool, found: bool = True):
        self._has_earnings = has_earnings
        self._found = found

    def get_8k_overview(self):
        if not self._found:
            return {"found": False}
        return {
            "found": True,
            "has_earnings": self._has_earnings,
            "items": ["2.02"] if self._has_earnings else ["5.02"],
            "content_type": "earnings" if self._has_earnings else "director_change",
            "context": "stub overview",
            "metadata": {},
        }


@pytest.mark.eval_unit
class TestDispatcherRouting:
    """_tool_analyze_latest_8k routes to exactly one analyzer based on the
    overview's ``has_earnings`` flag — proves the planner cannot end up with
    two contradictory 8-K analyses for the same filing."""

    def test_routes_to_earnings_when_has_earnings_true(self, monkeypatch):
        _seed(TICKER, "earnings_summary", _full_earnings_analysis())
        monkeypatch.setattr(
            sec_tools, "_get_shared_retriever",
            lambda ticker, header: _StubRetriever(has_earnings=True),
        )
        parsed = json.loads(sec_tools._tool_analyze_latest_8k(TICKER, llm=None))
        assert "key_metrics" in parsed, "Expected earnings shape (key_metrics)"
        assert "event_type" not in parsed, "Should not include material-event shape"

    def test_routes_to_material_event_when_has_earnings_false(self, monkeypatch):
        _seed(TICKER, "material_event_summary", _full_material_event_analysis())
        monkeypatch.setattr(
            sec_tools, "_get_shared_retriever",
            lambda ticker, header: _StubRetriever(has_earnings=False),
        )
        parsed = json.loads(sec_tools._tool_analyze_latest_8k(TICKER, llm=None))
        assert "event_type" in parsed, "Expected material-event shape (event_type)"
        assert "key_metrics" not in parsed, "Should not include earnings shape"

    def test_returns_json_when_no_filing_found(self, monkeypatch):
        monkeypatch.setattr(
            sec_tools, "_get_shared_retriever",
            lambda ticker, header: _StubRetriever(has_earnings=False, found=False),
        )
        parsed = json.loads(sec_tools._tool_analyze_latest_8k(TICKER, llm=None))
        assert parsed["found"] is False
