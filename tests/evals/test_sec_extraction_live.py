"""Live EDGAR extraction checks — requires network (no LLM keys).

Marker: network (excluded by default; run with ``-m network``)

Guards against edgartools/SEC regressions that make every narrative section
fail silently. edgartools 5.28.4 raised ``'NoneType' object has no attribute
'download'`` on all ``TenK[...]`` access; the retriever swallowed it into
``found: False`` and the Filings pipeline analysed the error text.
"""

import pytest

from agents.sec_workflow.get_SEC_data import SECDataRetrieval

TICKERS = ["AAPL", "BA"]


@pytest.fixture(scope="module", params=TICKERS)
def retriever(request):
    r = SECDataRetrieval(request.param)
    r.get_tenk_filing()
    r.get_tenq_filing()
    return r


@pytest.mark.network
class TestNarrativeSectionsExtract:
    def test_tenk_risk_factors(self, retriever):
        result = retriever.get_risk_factors_raw("10-K")
        assert result["found"], result["text"][:200]
        assert len(result["text"]) > 10_000

    def test_tenk_mda(self, retriever):
        result = retriever.get_mda_raw("10-K")
        assert result["found"], result["text"][:200]
        assert len(result["text"]) > 5_000

    def test_tenk_business(self, retriever):
        result = retriever.get_business_raw()
        assert result["found"], result["text"][:200]

    def test_tenq_mda(self, retriever):
        result = retriever.get_mda_raw("10-Q")
        assert result["found"], result["text"][:200]
        assert len(result["text"]) > 5_000

    def test_eightk_overview(self, retriever):
        result = retriever.get_8k_overview()
        assert result["found"], result.get("text", "")[:200]
