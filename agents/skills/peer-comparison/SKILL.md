---
name: peer-comparison
description: >
  Comparing the company with its competitors and industry: competitive
  landscape, market positioning, and the company's own valuation and growth
  metrics as the basis for the comparison.
tools:
  - analyze_competitors
  - get_industry_trends
  - get_stock_info
  - get_financial_metrics
  - web_search
---
# Peer comparison

Use this skill when the user asks how the company compares with competitors, who its competitors are, or where it stands in its industry.

## Tool selection

- `analyze_competitors`: the competitive landscape, market share and positioning. Start here.
- `get_industry_trends`: industry outlook and forecasts that frame the comparison.
- `get_stock_info`: the company's own valuation (P/E, price-to-book, market cap) to anchor the comparison.
- `get_financial_metrics`: the company's own revenue growth, net income growth and debt ratio.
- `web_search`: a targeted search for a specific peer figure the other tools didn't return.

All market and financial tools here are bound to {ticker}; peer figures come from research results. Don't present a peer's numbers as if they came from a tool that only covers {ticker}.

## Answering

- Compare on concrete dimensions (growth, profitability, valuation, market position) and cite where each figure comes from.
- If research tools are unavailable in this session, compare only on what the company's own data supports and say that competitor data isn't available.
