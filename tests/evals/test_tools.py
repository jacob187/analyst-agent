"""Parametrized tests calling each of the 12 SEC tools with AAPL.

Marker: eval_tools — requires GOOGLE_API_KEY (for SEC LLM tools) and
network access (SEC EDGAR + Yahoo Finance).

We split tools into categories:
- Content tools: return substantial text (>50 chars) with real financial data
- Meta tools: return non-empty strings (may be short availability listings)
- Special tools: have specific structural requirements (get_all_summaries, get_technical_analysis)
"""

import pytest


# Tools that return substantial content (>50 chars)
CONTENT_TOOLS = [
    "get_raw_risk_factors",
    "get_risk_factors_summary",
    "get_raw_management_discussion",
    "get_mda_summary",
    "get_balance_sheet_summary",
    "get_stock_price_history",
    "get_technical_analysis",
    "get_stock_info",
    "get_financial_metrics",
]

# Tools that may return short metadata/availability strings
META_TOOLS = [
    "get_raw_balance_sheets",
    "get_complete_10k_text",
]


@pytest.mark.eval_tools
class TestSecTools:
    """Validate each SEC tool returns usable output for AAPL."""

    @pytest.mark.parametrize("tool_name", CONTENT_TOOLS)
    def test_content_tool_returns_substantial_output(self, tools_dict, tool_name):
        """Content tools should return >50 chars of real data, no error strings.

        These tools hit real APIs (SEC EDGAR, Yahoo Finance) so they test the
        full data pipeline from retrieval through formatting. A short result
        or error string indicates a broken connection or parsing issue.
        """
        result = tools_dict[tool_name]("")
        assert isinstance(result, str)
        assert len(result) > 50, f"{tool_name} returned too little: {result[:100]}"
        assert "Failed to" not in result, f"{tool_name} returned error: {result[:200]}"

    @pytest.mark.parametrize("tool_name", META_TOOLS)
    def test_meta_tool_returns_nonempty(self, tools_dict, tool_name):
        """Meta tools should return a non-empty string."""
        result = tools_dict[tool_name]("")
        assert isinstance(result, str)
        assert len(result) > 0, f"{tool_name} returned empty string"
        assert "Failed to" not in result, f"{tool_name} returned error: {result[:200]}"

    def test_all_summaries_has_three_sections(self, tools_dict):
        """get_all_summaries combines risk, MD&A, and financial analysis.

        The output should contain section headers for all three, confirming
        the tool successfully aggregated data from three sub-tools.
        """
        result = tools_dict["get_all_summaries"]("")
        result_upper = result.upper()

        assert "RISK" in result_upper, "Missing RISK section in all_summaries"
        assert "MANAGEMENT" in result_upper or "OUTLOOK" in result_upper or "MD" in result_upper, (
            "Missing MANAGEMENT/OUTLOOK/MD&A section in all_summaries"
        )
        assert "FINANCIAL" in result_upper, "Missing FINANCIAL section in all_summaries"

    def test_technical_analysis_has_indicators(self, tools_dict):
        """get_technical_analysis should include RSI, MACD, and moving averages.

        These are the core technical indicators. If any are missing, the
        TechnicalIndicators calculator or the output formatter has a bug.
        """
        result = tools_dict["get_technical_analysis"]("")
        result_upper = result.upper()

        assert "RSI" in result_upper, "Missing RSI in technical analysis"
        assert "MACD" in result_upper, "Missing MACD in technical analysis"
        assert "MA" in result_upper or "MOVING AVERAGE" in result_upper, (
            "Missing moving averages in technical analysis"
        )
