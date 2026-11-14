---
name: news-research
description: >
  Current news and web research: latest headlines and developments, general
  web search and fact-checking, in-depth multi-source research, and industry
  trends and forecasts.
tools:
  - get_company_news
  - web_search
  - deep_research
  - get_industry_trends
  - analyze_competitors
---
# News and web research

Use this skill for recent news, current events, anything that happened after the latest filing, and questions that need the open web.

## Tool selection

- `get_company_news`: latest headlines, news summaries and source links. First choice for "what's the latest news".
- `web_search`: a targeted search for current information or fact-checking. The input is a search query string; make it specific.
- `deep_research`: comprehensive multi-source research on one topic. It is slow and expensive; use it only for in-depth questions a single search won't answer. The input is the research topic.
- `get_industry_trends`: industry trends, outlook and forecasts.
- `analyze_competitors`: competitive landscape and market positioning (for structured peer comparisons, load the peer-comparison skill).

## Answering

- News is only half the picture: also check how the price and fundamentals reacted (technicals skill), and the relevant filing if the news refers to one.
- Attribute claims to their sources and give dates; news can be stale or wrong.
- If news describes an SEC filing (for example an 8-K), the filing is authoritative. Describe the event once, and if the news and the filing disagree, say so explicitly.
- If a research tool reports that web research is unavailable (for example the daily search limit is reached), say so and answer from SEC filings and market data instead.
