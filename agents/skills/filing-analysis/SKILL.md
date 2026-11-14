---
name: filing-analysis
description: >
  10-K / 10-Q filing analysis: risk factors, MD&A and management outlook,
  balance sheet health, business overview, cybersecurity disclosure, legal
  proceedings, and fundamental growth/debt metrics.
tools:
  - get_risk_factors_summary
  - get_mda_summary
  - get_balance_sheet_summary
  - get_all_summaries
  - get_raw_risk_factors
  - get_raw_management_discussion
  - get_raw_balance_sheets
  - get_complete_10k_text
  - get_business_overview
  - get_cybersecurity_disclosure
  - get_legal_proceedings
  - get_financial_metrics
---
# Filing analysis (10-K / 10-Q)

Use this skill for risks, management outlook, financial statements and other periodic-filing content.

## Tool selection

- `get_risk_factors_summary`: structured analysis of risks from the most recent filing (10-Q preferred, 10-K fallback): sentiment, key risks, severity. First choice for "what are the main risks".
- `get_mda_summary`: management outlook, sentiment and future plans from the most recent MD&A (10-Q preferred).
- `get_balance_sheet_summary`: financial health, key metrics and red flags.
- `get_all_summaries`: a comprehensive overview across risks, MD&A and financials in one call. Use it for broad "give me an overview / due diligence" questions instead of calling the three summaries separately.
- `get_financial_metrics`: year-over-year revenue growth, net income growth and debt-to-assets from the income statement and balance sheet (market data, not a filing). Use it for growth and leverage numbers.
- `get_business_overview`: the business description, products, segments and markets (10-K Item 1).
- `get_cybersecurity_disclosure`: cybersecurity risk management and governance (10-K Item 1C).
- `get_legal_proceedings`: significant pending litigation and regulatory actions (10-K Item 3).
- Raw text tools (`get_raw_risk_factors`, `get_raw_management_discussion`, `get_raw_balance_sheets`, `get_complete_10k_text`): only when the user asks for the original wording or a detail the structured summaries don't cover. `get_complete_10k_text` lists which 10-K sections are retrievable as raw text.

## Answering

- Say which filing (10-K or 10-Q) and period the data comes from when the tool reports it.
- Separate what the filing states from your interpretation.
- Quote the specific figures (revenue, margins, debt, growth rates) with their filing type and period.
- Filings are backward-looking. Also load the news-research and technicals skills to check what happened since the filing (recent news) and how the stock is trading (price, valuation), and add that as context unless the question is purely about the filing text.
- For investment-decision questions, make sure the answer covers fundamentals, risks, technicals and recent developments.
