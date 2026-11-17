"""Test fakes for the SEC analysis cache: an in-memory EDGAR (edgartools-shaped
``Company`` / filing / TenK / TenQ / EightK objects) and a chat model that
counts calls at the LLM boundary.

``FakeEdgar.install(monkeypatch)`` swaps edgartools' ``Company`` inside
``agents.sec_workflow.get_SEC_data``, so the real ``SECDataRetrieval`` runs
against these objects with no network. ``FakeEdgar.requests`` counts filing
index lookups; ``FakeEdgar.companies`` counts ``Company`` builds (each one an
EDGAR submissions request in production).

Like EDGAR, a company's submissions are a recent page (the newest
``FakeEdgar.recent_size`` filings, fetched with the ``Company``) plus older
pages that edgartools downloads only on a full load (``get_filings`` with
``trigger_full_load=True``, its default); ``FakeEdgar.page_downloads`` counts
those older-page requests.
"""

from __future__ import annotations

import asyncio
import threading
import time
from collections import Counter
from dataclasses import dataclass, field
from datetime import date
from typing import Any

import pandas as pd
from langchain_core.language_models.chat_models import BaseChatModel
from langchain_core.messages import AIMessage
from langchain_core.outputs import ChatGeneration, ChatResult
from pydantic import Field

from agents.sec_workflow.sec_llm_models import (
    BalanceSheetAnalysis,
    BusinessOverviewAnalysis,
    CashFlowAnalysis,
    CybersecurityAnalysis,
    EarningsAnalysis,
    IncomeStatementAnalysis,
    LegalProceedingsAnalysis,
    MarketRiskAnalysis,
    MaterialEventAnalysis,
    MDnAAnalysis,
    RiskFactorAnalysis,
)

CIK = 320193
COMPANY_NAME = "TESTCO INC"


# ── edgartools-shaped filing objects ─────────────────────────────────────────


class _Statement:
    def __init__(self, frame: pd.DataFrame):
        self._frame = frame

    def to_dataframe(self) -> pd.DataFrame:
        return self._frame


class _Financials:
    """``TenK.financials`` / ``TenQ.financials``: XBRL statements."""

    def __init__(self, balance_sheet: pd.DataFrame | None):
        self._balance = balance_sheet

    def balance_sheet(self):
        return _Statement(self._balance) if self._balance is not None else None

    def income_statement(self):
        return None

    def cashflow_statement(self):
        return None


class FakeReport:
    """TenK / TenQ: ``report["Item 1A"]`` → section text (None if absent).

    Like edgartools, ``sections`` parses the document on first access and
    caches it with an unlocked check-then-set, and ``__getitem__`` reads
    through ``sections``. Instrumentation for concurrency tests:

    - ``parses``: how many times the document was parsed.
    - ``reads[key]``: section extractions attempted per key.
    - ``parse_delay`` / ``read_delay``: seconds a parse / extraction blocks,
      so concurrent callers overlap.
    - ``gates[key]``: extraction of ``key`` sets ``entered[key]`` then waits
      for the gate.
    - ``failures[key]``: extraction of ``key`` raises this many times first
      (a transient EDGAR / parse error).
    - ``blanks[key]``: extraction of ``key`` returns None this many times
      first (edgartools finds no sections when the filing HTML didn't load).
    """

    def __init__(self, sections: dict[str, str], balance_sheet: pd.DataFrame | None):
        self._sections = sections
        self.financials = _Financials(balance_sheet)
        self._parsed: dict[str, str] | None = None
        self._count_lock = threading.Lock()
        self.parses = 0
        self.reads: Counter[str] = Counter()
        self.parse_delay = 0.0
        self.read_delay = 0.0
        self.gates: dict[str, threading.Event] = {}
        self.entered: dict[str, threading.Event] = {}
        self.failures: dict[str, int] = {}
        self.blanks: dict[str, int] = {}

    @property
    def sections(self) -> dict[str, str]:
        if self._parsed is None:
            with self._count_lock:
                self.parses += 1
            if self.parse_delay:
                time.sleep(self.parse_delay)
            self._parsed = dict(self._sections)
        return self._parsed

    def __getitem__(self, key: str) -> str | None:
        sections = self.sections
        with self._count_lock:
            self.reads[key] += 1
            fail = self.failures.get(key, 0) > 0
            if fail:
                self.failures[key] -= 1
            blank = self.blanks.get(key, 0) > 0
            if blank:
                self.blanks[key] -= 1
        if key in self.gates:
            self.entered.setdefault(key, threading.Event()).set()
            self.gates[key].wait()
        if self.read_delay:
            time.sleep(self.read_delay)
        if fail:
            raise RuntimeError(f"transient failure extracting {key}")
        if blank:
            return None
        return sections.get(key)


class _Earnings:
    """``EightK.earnings`` (edgartools ``EarningsRelease``) without tables."""

    detected_scale = "millions"
    income_statement = None
    balance_sheet = None
    cash_flow_statement = None

    def __init__(self, context: str):
        self._context = context

    def to_context(self, detail: str = "standard") -> str:
        return self._context


class FakeEightK:
    def __init__(self, items: dict[str, str], has_earnings: bool, filed: str):
        self._items = items
        self.items = list(items)
        self.has_earnings = has_earnings
        self.has_press_release = has_earnings
        self.content_type = "earnings" if has_earnings else "director_change"
        self.is_amendment = False
        self.date_of_report = filed
        self.earnings = _Earnings(f"Earnings release filed {filed}") if has_earnings else None
        self._count_lock = threading.Lock()
        self.reads: Counter[str] = Counter()
        self.read_delay = 0.0
        # Called (once) when the overview is built, e.g. to land a concurrent
        # refresh between the overview read and the analysis-input read.
        self.after_overview = None

    def to_context(self, detail: str = "standard") -> str:
        hook, self.after_overview = self.after_overview, None
        if hook is not None:
            hook()
        return f"8K: {COMPANY_NAME} Current Report\nItems: {', '.join(self.items)}"

    def __getitem__(self, item: str) -> str | None:
        with self._count_lock:
            self.reads[item] += 1
        if self.read_delay:
            time.sleep(self.read_delay)
        return self._items.get(item)


@dataclass
class FakeFiling:
    """``EntityFiling``: the attributes ``SECDataRetrieval`` reads + ``obj()``."""

    form: str
    accession_number: str
    filing_date: date
    report: Any
    period_of_report: str = "2026-09-30"
    cik: int = CIK
    company: str = COMPANY_NAME
    obj_calls: int = 0  # each one builds a fresh TenK/TenQ in edgartools
    obj_delay: float = 0.0  # seconds obj() blocks, so racing callers overlap

    def obj(self):
        self.obj_calls += 1
        if self.obj_delay:
            time.sleep(self.obj_delay)
        return self.report


def _balance_frame(total_assets: float) -> pd.DataFrame:
    return pd.DataFrame(
        {"2026-09-30": [total_assets, total_assets * 0.4]},
        index=["Total Assets", "Total Liabilities"],
    )


def tenk(accession: str, filed: str, risk: str, mda: str = "10-K MD&A text") -> FakeFiling:
    report = FakeReport({"Item 1A": risk, "Item 7": mda}, _balance_frame(352e9))
    return FakeFiling("10-K", accession, date.fromisoformat(filed), report)


def tenq(
    accession: str, filed: str, risk: str, mda: str = "10-Q MD&A text", *, balance_sheet: bool = True,
) -> FakeFiling:
    balance = _balance_frame(360e9) if balance_sheet else None
    report = FakeReport({"Item 1A": risk, "Item 2": mda}, balance)
    return FakeFiling("10-Q", accession, date.fromisoformat(filed), report)


def eightk(accession: str, filed: str, *, earnings: bool = True, item_text: str = "Item text") -> FakeFiling:
    items = {"2.02": item_text} if earnings else {"5.02": item_text}
    return FakeFiling("8-K", accession, date.fromisoformat(filed), FakeEightK(items, earnings, filed))


class FakeTwentyF:
    """edgartools ``TwentyF``: named section accessors, no ``Item X`` keys."""

    def __init__(self, risk: str, mda: str, business: str):
        self.risk_factors = risk
        self.management_discussion = mda
        self.business = business


def twentyf(accession: str, filed: str, risk: str, mda: str = "20-F MD&A text") -> FakeFiling:
    report = FakeTwentyF(risk, mda, "20-F business text")
    return FakeFiling("20-F", accession, date.fromisoformat(filed), report)


class _Filings:
    def __init__(self, filings: list[FakeFiling]):
        self._filings = filings

    def __len__(self) -> int:
        return len(self._filings)

    def latest(self, n: int = 1):
        return self._filings[0] if self._filings else None


@dataclass
class FakeEdgar:
    """In-memory EDGAR for one company. ``filings[form]`` is newest-first."""

    filings: dict[str, list[FakeFiling]] = field(default_factory=dict)
    requests: int = 0
    latency: float = 0.0  # seconds each filing-index lookup blocks for
    # ``Company(ticker)`` builds: each one downloads the submissions JSON.
    companies: int = 0
    # When set, ``Company(ticker)`` raises it (EDGAR 429 / 5xx / outage).
    outage: Exception | None = None
    # ``Company(ticker)`` builds that hit the outage (each a failed request).
    failed_companies: int = 0
    # Newest filings (all forms) on the recent submissions page; None: all.
    recent_size: int | None = None
    # Older submissions pages one full load downloads (edgartools ``files``).
    older_pages: int = 2
    page_downloads: int = 0

    def publish(self, filing: FakeFiling) -> None:
        self.filings.setdefault(filing.form, []).insert(0, filing)

    def recent(self, filings: list[FakeFiling]) -> list[FakeFiling]:
        """The ``filings`` that are on the recent submissions page."""
        if self.recent_size is None:
            return filings
        newest = sorted(
            (f for fs in self.filings.values() for f in fs),
            key=lambda f: f.filing_date, reverse=True,
        )[: self.recent_size]
        return [f for f in filings if any(f is n for n in newest)]

    def company(self, ticker: str) -> "_FakeCompany":
        if self.outage is not None:
            self.failed_companies += 1
            raise self.outage
        self.companies += 1
        return _FakeCompany(self, ticker)

    def install(self, monkeypatch) -> "FakeEdgar":
        from agents.sec_workflow import get_SEC_data

        monkeypatch.setattr(get_SEC_data, "Company", self.company)
        return self


class _FakeCompany:
    """edgartools ``Company``: ``latest(form)``, ``get_filings(form)``."""

    is_foreign = False

    def __init__(self, edgar: FakeEdgar, ticker: str):
        self._edgar = edgar
        self.ticker = ticker
        self.name = COMPANY_NAME
        self._loaded_all_filings = False

    def latest(self, form: str, n: int = 1):
        # edgartools: the recent page first, a full load only if it has none.
        filings = self.get_filings(form, trigger_full_load=False)
        if len(filings) < n:
            filings = self.get_filings(form, trigger_full_load=True)
        return filings.latest(n)

    def get_filings(self, form: str, trigger_full_load: bool = True, **_kwargs):
        self._edgar.requests += 1
        if self._edgar.latency:
            time.sleep(self._edgar.latency)
        if trigger_full_load and not self._loaded_all_filings:
            self._edgar.page_downloads += self._edgar.older_pages
            self._loaded_all_filings = True
        filings = self._edgar.filings.get(form, [])
        if not self._loaded_all_filings:
            filings = self._edgar.recent(filings)
        return _Filings(filings)


# ── LLM boundary ─────────────────────────────────────────────────────────────


# Each analysis schema's field that no other schema has → a valid answer
# (``n`` numbers the call, so tests can tell answers apart).
_ANSWERS: dict[str, Any] = {
    '"key_risks"': lambda n: RiskFactorAnalysis(
        summary=f"risk analysis #{n}", key_risks=["Supply chain"],
        risk_categories={"operational": ["Supply chain"]}, sentiment_score=-2.0,
        sentiment_analysis="Moderate risk.", form_type="10-K",
    ),
    '"future_outlook"': lambda n: MDnAAnalysis(
        summary=f"mda analysis #{n}", key_points=["Growth"], financial_highlights=["Revenue up"],
        future_outlook="Positive.", sentiment_score=5.0, sentiment_analysis="Upbeat.",
        form_type="10-K",
    ),
    '"liquidity_analysis"': lambda n: BalanceSheetAnalysis(
        summary=f"balance analysis #{n}", key_metrics=["Assets $352B"],
        liquidity_analysis="Strong.", solvency_analysis="Low leverage.",
        growth_trends="Steady.", financial_highlights=["Cash $60B"], red_flags=[],
    ),
    '"beats_misses"': lambda n: EarningsAnalysis(
        summary=f"earnings analysis #{n}", key_metrics=["EPS $2.40"], beats_misses=["Beat"],
        sentiment_score=6.0, sentiment_analysis="Strong quarter.",
    ),
    '"impact_assessment"': lambda n: MaterialEventAnalysis(
        summary=f"event analysis #{n}", event_type="leadership_change", key_points=["CEO change"],
        impact_assessment="Short-term volatility.", sentiment_score=-1.0,
        sentiment_analysis="Mildly negative.",
    ),
    '"business_segments"': lambda n: BusinessOverviewAnalysis(
        summary=f"business analysis #{n}", business_segments=["Devices"],
        key_products_services=["Phones"], competitive_position="Strong brand.",
        sentiment_score=4.0, sentiment_analysis="Solid.",
    ),
    '"governance_overview"': lambda n: CybersecurityAnalysis(
        summary=f"cybersecurity analysis #{n}", governance_overview="Board audit committee.",
        key_disclosures=["NIST CSF"], red_flags=[], sentiment_score=2.0,
        sentiment_analysis="Adequate.",
    ),
    '"key_cases"': lambda n: LegalProceedingsAnalysis(
        summary=f"legal analysis #{n}", key_cases=["Antitrust suit"], red_flags=[],
        overall_risk="Moderate.", sentiment_score=-1.0, sentiment_analysis="Manageable.",
    ),
    '"key_exposures"': lambda n: MarketRiskAnalysis(
        summary=f"market risk analysis #{n}", key_exposures=["FX"],
        risk_assessment="Hedged.", red_flags=[], sentiment_score=0.0,
        sentiment_analysis="Neutral.",
    ),
    '"revenue_analysis"': lambda n: IncomeStatementAnalysis(
        summary=f"income statement analysis #{n}", key_metrics=["Revenue $400B"],
        revenue_analysis="Growing.", profitability_analysis="Stable margins.", red_flags=[],
        outlook="Positive.", sentiment_score=5.0, sentiment_analysis="Healthy.",
    ),
    '"operating_cash_flow_analysis"': lambda n: CashFlowAnalysis(
        summary=f"cash flow analysis #{n}", key_metrics=["FCF $100B"],
        operating_cash_flow_analysis="Strong.", free_cash_flow_analysis="Rising.",
        red_flags=[], outlook="Flexible.", sentiment_score=5.0, sentiment_analysis="Healthy.",
    ),
}


def _response_for(prompt: str, n: int) -> str:
    """A valid structured answer for whichever analysis the prompt asks for
    (picked by the schema field names in the format instructions)."""
    for marker, answer in _ANSWERS.items():
        if marker in prompt:
            return answer(n).model_dump_json()
    raise AssertionError("fake LLM got a prompt for an analysis it can't answer")


class CountingChatModel(BaseChatModel):
    """Answers SEC analysis prompts and records every prompt it was sent.

    Async only: a sync ``invoke`` means an SEC analysis skipped ``ainvoke``
    (and with it the chat run's callbacks), which these tests treat as a bug.
    """

    prompts: list[str] = Field(default_factory=list)
    delay: float = 0.0
    in_flight: int = 0
    peak_in_flight: int = 0  # most calls ever awaiting an answer at once

    @property
    def _llm_type(self) -> str:
        return "counting-fake"

    @property
    def calls(self) -> int:
        return len(self.prompts)

    @property
    def analyses(self) -> list[str]:
        """The analysis each prompt asked for, named by its schema's
        distinguishing field (e.g. "key_risks" for risk factors)."""
        return [
            next((m.strip('"') for m in _ANSWERS if m in p), "unknown") for p in self.prompts
        ]

    def _generate(self, messages, stop=None, run_manager=None, **kwargs) -> ChatResult:
        raise AssertionError("SEC analyses must call the LLM via ainvoke")

    async def _agenerate(self, messages, stop=None, run_manager=None, **kwargs) -> ChatResult:
        prompt = "\n".join(str(m.content) for m in messages)
        self.prompts.append(prompt)
        self.in_flight += 1
        self.peak_in_flight = max(self.peak_in_flight, self.in_flight)
        try:
            if self.delay:
                await asyncio.sleep(self.delay)
        finally:
            self.in_flight -= 1
        message = AIMessage(content=_response_for(prompt, len(self.prompts)))
        return ChatResult(generations=[ChatGeneration(message=message)])
