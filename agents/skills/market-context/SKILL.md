---
name: market-context
description: >
  Broad market and macro context plus past daily briefings: S&P 500, Nasdaq,
  Dow, VIX, Treasury yields, the dollar, and the user's briefing history and
  latest morning briefing.
tools:
  - get_market_overview
  - get_macro_indicators
  - get_briefing_history
  - get_latest_briefing
---
# Market context and briefing history

Use this skill for "how is the market doing", interest rates and the macro environment, benchmark comparisons, and questions about past daily briefings.

## Tool selection

- `get_market_overview`: current levels and daily changes for the S&P 500, Nasdaq, Dow Jones and VIX. Use it for broad market context or to compare the stock against the market.
- `get_macro_indicators`: 10-year and 5-year Treasury yields, VIX and the US Dollar Index with 1-month changes. Use it for interest-rate and macro questions.
- `get_briefing_history`: recent daily briefing analyses for this ticker (outlook trends, signals, alerts). The input is the number of days to look back (default 7).
- `get_latest_briefing`: the most recent full morning briefing: market regime, all tickers and alerts.

These tools are not about the company's own filings or price; pair this skill with technicals or filing analysis when the question is about how the market backdrop affects the stock.

## Answering

- Quote the levels and changes the tools return, and say what they imply for the question (risk-on vs risk-off, rate pressure on valuations, and so on).
- Briefing history needs a signed-in user. If the tool reports no history or no user context, say that briefing history isn't available rather than inventing one.
