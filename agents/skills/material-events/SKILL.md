---
name: material-events
description: >
  8-K filings and material events: the latest earnings release, leadership
  changes, material agreements, cybersecurity incidents, Reg FD disclosures,
  and the raw text of a specific 8-K item.
tools:
  - analyze_latest_8k
  - get_8k_item
---
# 8-K / material events

Use this skill for any question about the company's most recent 8-K, an earnings release, or a material event (executive departures or appointments, material agreements, cybersecurity incidents, Reg FD disclosures).

## Tool selection

- `analyze_latest_8k` is the analysis tool for ANY question about the most recent 8-K. It returns structured JSON (sentiment, key points, impact assessment, event type and filing metadata). It routes internally: an Item 2.02 earnings release with a press release goes to earnings analysis; everything else goes to material event analysis. Call it at most once per question; there is exactly one 8-K analysis tool, so never try to analyse the same filing a second way.
- `get_8k_item` returns the raw text of one 8-K item. Pass the item number as the input (for example `2.02` for earnings, `1.01` for material agreements, `5.02` for leadership changes). Use it only when the user needs the original wording of a specific item or to verify a detail; for analysis use `analyze_latest_8k`.
- If `analyze_latest_8k` reports that no 8-K was found, say so plainly instead of guessing.

## Cross-checking (one filing, one description)

Several sources can describe the same filing: the 8-K analysis, the raw item text, and news or research results if those skills are loaded. Treat them as describing one event:

- Identify the filing by its accession number, or by form type and filing date, from `filing_metadata`.
- Describe each filing exactly once. Never write two separate sections that characterise the same filing differently.
- Compare structured fields across sources, especially the event type (for example earnings vs. leadership change). If sources disagree, say so explicitly: name the filing, the sources and the conflicting values, and state which you rely on and why (the filing text outranks news coverage).

## Answering

- Lead with what happened, the filing date and the item(s) it covers, then the impact assessment.
- For earnings releases, report the headline figures the analysis returns and how management characterised them.
