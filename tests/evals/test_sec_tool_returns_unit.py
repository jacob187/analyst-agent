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
model. The tests seed the ``filing_analyses`` table (a real temp SQLite) with
a hand-crafted ``model_dump()`` for the latest filing of an in-memory EDGAR
(tests/sec_fakes.py), so the tool's return path runs without any LLM call.
These are fakes, not mocks of project code — see CLAUDE.md testing philosophy.
"""

import json
import pytest

import api.db as db_module
from agents.tools import sec_tools
from agents.sec_workflow.sec_llm_models import (
    EarningsAnalysis,
    MaterialEventAnalysis,
    MDnAAnalysis,
    RiskFactorAnalysis,
    BalanceSheetAnalysis,
)
from tests.sec_fakes import FakeEdgar, eightk, tenk, tenq


TICKER = "TESTCO"
TENK_ACC = "0000320193-25-000101"
TENQ_ACC = "0000320193-26-000111"
EIGHTK_ACC = "0000320193-26-000121"

# Which (form, accession, analysis_type) row each tool reads for the fake
# EDGAR below: risk falls back to the 10-K (10-Q risk text is boilerplate),
# MD&A uses the substantive 10-Q.
_ROW = {
    "risk": ("10-K", TENK_ACC, "risk_10k"),
    "mda": ("10-Q", TENQ_ACC, "mda_10q"),
    "balance": ("10-K", TENK_ACC, "balance"),
    "earnings": ("8-K", EIGHTK_ACC, "earnings"),
    "event": ("8-K", EIGHTK_ACC, "event"),
}


@pytest.fixture(autouse=True)
async def fresh_db(tmp_path, monkeypatch):
    monkeypatch.setattr(db_module, "DB_PATH", tmp_path / "test.db")
    monkeypatch.setattr(db_module, "_db", None)
    await db_module.init_db()
    yield
    await db_module.close_db()


def _edgar(monkeypatch, *, has_earnings: bool = True, has_8k: bool = True) -> FakeEdgar:
    fake = FakeEdgar()
    fake.publish(tenk(TENK_ACC, "2025-11-01", "10-K risk factors text"))
    fake.publish(tenq(
        TENQ_ACC, "2026-02-01",
        "There have been no material changes to the risk factors previously disclosed.",
        mda="Services revenue grew 12% year over year.",
    ))
    if has_8k:
        fake.publish(eightk(EIGHTK_ACC, "2026-04-07", earnings=has_earnings))
    return fake.install(monkeypatch)


@pytest.fixture(autouse=True)
def edgar(monkeypatch):
    return _edgar(monkeypatch)


async def _seed(_ticker: str, key: str, model: object) -> None:
    """Seed ``filing_analyses`` with a Pydantic model's ``model_dump()``."""
    form, accession, analysis_type = _ROW[key]
    await db_module.save_filing_analysis(
        TICKER, form, accession, analysis_type, json.dumps(model.model_dump()),
    )


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


async def _earnings_8k() -> dict:
    return json.loads(await sec_tools._tool_analyze_latest_8k(TICKER, llm=None))


@pytest.mark.eval_unit
class TestEarningsSummaryReturnsFullModel:
    """analyze_latest_8k (earnings 8-K) returns JSON containing every EarningsAnalysis field."""

    async def test_returns_valid_json(self):
        await _seed(TICKER, "earnings", _full_earnings_analysis())
        assert isinstance(await _earnings_8k(), dict)

    async def test_preserves_all_pydantic_fields(self):
        await _seed(TICKER, "earnings", _full_earnings_analysis())
        parsed = await _earnings_8k()
        for field in EarningsAnalysis.model_fields:
            assert field in parsed, f"Tool dropped field {field!r}"

    async def test_preserves_all_key_metrics(self):
        """The old f-string truncated key_metrics to 5 — must now keep all 8."""
        model = _full_earnings_analysis()
        await _seed(TICKER, "earnings", model)
        parsed = await _earnings_8k()
        assert parsed["key_metrics"] == model.key_metrics
        assert len(parsed["key_metrics"]) == 8

    async def test_preserves_sentiment_analysis(self):
        """The old f-string dropped sentiment_analysis entirely — proves it returns now."""
        model = _full_earnings_analysis()
        await _seed(TICKER, "earnings", model)
        parsed = await _earnings_8k()
        assert parsed["sentiment_analysis"] == model.sentiment_analysis


@pytest.mark.eval_unit
class TestMaterialEventSummaryReturnsFullModel:
    """analyze_latest_8k (non-earnings 8-K) returns JSON containing every field."""

    @pytest.fixture(autouse=True)
    def edgar(self, monkeypatch):
        return _edgar(monkeypatch, has_earnings=False)

    async def test_returns_valid_json(self):
        await _seed(TICKER, "event", _full_material_event_analysis())
        assert isinstance(await _earnings_8k(), dict)

    async def test_preserves_all_pydantic_fields(self):
        await _seed(TICKER, "event", _full_material_event_analysis())
        parsed = await _earnings_8k()
        for field in MaterialEventAnalysis.model_fields:
            assert field in parsed, f"Tool dropped field {field!r}"

    async def test_preserves_event_type_and_filing_metadata(self):
        """These two fields are what makes cross-tool reconciliation possible —
        they identify which filing the analysis describes."""
        await _seed(TICKER, "event", _full_material_event_analysis())
        parsed = await _earnings_8k()
        assert parsed["event_type"] == "executive_compensation"
        assert parsed["filing_metadata"]["filing_date"] == "2026-04-07"
        assert parsed["filing_reference"]["accession"] == EIGHTK_ACC


@pytest.mark.eval_unit
class TestMDASummaryReturnsFullModel:
    """_tool_mda_summary returns JSON containing every MDnAAnalysis field —
    crucially ``financial_highlights`` (where dollar amounts live) which the
    old f-string return dropped entirely."""

    async def test_preserves_all_pydantic_fields(self):
        await _seed(TICKER, "mda", _full_mda_analysis())
        parsed = json.loads(await sec_tools._tool_mda_summary(TICKER, llm=None))
        for field in MDnAAnalysis.model_fields:
            assert field in parsed, f"Tool dropped field {field!r}"

    async def test_preserves_financial_highlights(self):
        """``financial_highlights`` is where Diluted EPS / Operating Income /
        capex numbers live for MD&A. The old f-string dropped the entire field."""
        model = _full_mda_analysis()
        await _seed(TICKER, "mda", model)
        parsed = json.loads(await sec_tools._tool_mda_summary(TICKER, llm=None))
        assert parsed["financial_highlights"] == model.financial_highlights
        assert len(parsed["financial_highlights"]) == 4


@pytest.mark.eval_unit
class TestRiskSummaryReturnsFullModel:
    """_tool_risk_factors_summary returns JSON containing the categorized
    ``risk_categories`` dict — which the old f-string return dropped."""

    async def test_preserves_all_pydantic_fields(self):
        await _seed(TICKER, "risk", _full_risk_analysis())
        parsed = json.loads(await sec_tools._tool_risk_factors_summary(TICKER, llm=None))
        for field in RiskFactorAnalysis.model_fields:
            assert field in parsed, f"Tool dropped field {field!r}"

    async def test_preserves_risk_categories(self):
        model = _full_risk_analysis()
        await _seed(TICKER, "risk", model)
        parsed = json.loads(await sec_tools._tool_risk_factors_summary(TICKER, llm=None))
        assert parsed["risk_categories"] == model.risk_categories


@pytest.mark.eval_unit
class TestBalanceSheetSummaryReturnsFullModel:
    """_tool_balance_sheet_summary returns JSON containing the four substantive
    fields the old f-string dropped: ``liquidity_analysis``, ``solvency_analysis``,
    ``growth_trends``, ``financial_highlights``."""

    async def test_preserves_all_pydantic_fields(self):
        await _seed(TICKER, "balance", _full_balance_sheet_analysis())
        parsed = json.loads(await sec_tools._tool_balance_sheet_summary(TICKER, llm=None))
        for field in BalanceSheetAnalysis.model_fields:
            assert field in parsed, f"Tool dropped field {field!r}"

    async def test_preserves_liquidity_solvency_growth(self):
        model = _full_balance_sheet_analysis()
        await _seed(TICKER, "balance", model)
        parsed = json.loads(await sec_tools._tool_balance_sheet_summary(TICKER, llm=None))
        assert parsed["liquidity_analysis"] == model.liquidity_analysis
        assert parsed["solvency_analysis"] == model.solvency_analysis
        assert parsed["growth_trends"] == model.growth_trends


# ── Dispatcher routing tests ────────────────────────────────────────────────


@pytest.mark.eval_unit
class TestDispatcherRouting:
    """_tool_analyze_latest_8k routes to exactly one analyzer based on the
    overview's ``has_earnings`` flag — proves the planner cannot end up with
    two contradictory 8-K analyses for the same filing. Both analyses are
    cached, so the routing alone decides which one is served."""

    async def _seed_both(self):
        await _seed(TICKER, "earnings", _full_earnings_analysis())
        await _seed(TICKER, "event", _full_material_event_analysis())

    async def test_routes_to_earnings_when_has_earnings_true(self, monkeypatch):
        _edgar(monkeypatch, has_earnings=True)
        await self._seed_both()
        parsed = await _earnings_8k()
        assert "key_metrics" in parsed, "Expected earnings shape (key_metrics)"
        assert "event_type" not in parsed, "Should not include material-event shape"

    async def test_routes_to_material_event_when_has_earnings_false(self, monkeypatch):
        _edgar(monkeypatch, has_earnings=False)
        await self._seed_both()
        parsed = await _earnings_8k()
        assert "event_type" in parsed, "Expected material-event shape (event_type)"
        assert "key_metrics" not in parsed, "Should not include earnings shape"

    async def test_returns_json_when_no_filing_found(self, monkeypatch):
        _edgar(monkeypatch, has_8k=False)
        parsed = await _earnings_8k()
        assert parsed["found"] is False
