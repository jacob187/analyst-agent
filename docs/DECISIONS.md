# Decisions

Architecture and product decisions behind Analyst Agent, with the evidence for each. Newest first. The full benchmark tables are in [PR #57](https://github.com/jacob187/analyst-agent/pull/57); the harness that produced them is `evals/benchmark/`.

All benchmark numbers come from the same 26 questions on AAPL, NVDA and JPM (filing lookups, material events, price and technicals, research, multi-step), run on 2026-11-14 with each run in its own empty database.

---

## 1. Keep the planner graph; reject the skills-based agent

**Context.** The agent is a LangGraph planner: a router, a planner that fans steps out to parallel tool workers, a reconciler and a synthesizer. A single tool-calling agent that loads task instructions ("skills") on demand promised to be cheaper and faster.

**Options.** Keep the planner graph, or replace it with the skills agent.

**Decision.** Keep the planner graph. The skills agent was built, measured and removed.

**Evidence.** Same model (Gemini 3.1 Flash-Lite) for every run, so only the architecture differs.

| | Planner graph | Skills agent | Skills agent, every tool + completeness prompt |
|---|---:|---:|---:|
| Cost per question | $0.00707 | $0.00548 (−22%) | $0.00726 (+3%) |
| Seconds per question | 12.3 | 7.0 (−43%) | 7.5 |
| Questions won, Opus 5 judge | 20 | 5 | 6 (vs 20) |
| Questions won, Sonnet 5.5 judge | — | — | 8 (vs 18) |

**Trade-offs.** We keep a slower, more expensive architecture because its answers were better on three out of four judgements. The fourth judgement favoured the skills agent 13–8, but it was discarded: see decision 3. Giving the skills agent every tool made it as expensive as the planner graph without closing the quality gap. A cost optimisation without a quality gate would have shipped a worse product.

**Caveat.** These runs were measured while planner steps passed no arguments to their tools ([#59](https://github.com/jacob187/analyst-agent/issues/59)), so the planner graph won with a handicap. The comparison will be re-run once steps pass arguments.

---

## 2. Claude Haiku 5.5 as the default and free-trial model

**Context.** The anonymous free trial and the default chat model run on the operator's key, so the choice sets both answer quality and the bill.

**Options.** Gemini 3.1 Flash-Lite (the previous default), Gemini 3.6 Flash, Gemini 3.8 Flash, Claude Haiku 5.5, Claude Sonnet 5.5. GPT-5.4 mini was excluded after every question failed on an exhausted OpenAI quota.

**Decision.** Claude Haiku 5.5 is the default and the free-trial model. Gemini 3.6 Flash and Gemini 3.1 Flash-Lite stay available on the operator's key; everything else needs the user's own key.

**Evidence.** Each model was judged against Flash-Lite on the planner graph.

| Model | $ / question | Sonnet 5.5 judge W–T–L | Gemini 3.1 Pro judge W–T–L | Unique factual errors (Sonnet / Gemini Pro) |
|---|---:|---|---|---|
| Gemini 3.1 Flash-Lite (reference) | $0.00698 | — | — | 22 / 4 |
| **Claude Haiku 5.5** | **$0.00761** | **26–0–0** | **23–0–3** | **0 / 2** |
| Gemini 3.8 Flash | $0.03068 | 24–0–1 | 19–1–5 | 5 / 1 |
| Gemini 3.6 Flash | $0.03087 | 22–0–3 | not re-judged | 4 / — |
| Claude Sonnet 5.5 | $0.13668 | 26–0–0 | not re-judged | 0 / — |

Haiku matched Sonnet 5.5's quality at about 1/18 of the price and cost about 1.1× Flash-Lite with far fewer factual errors under both judges.

**Trade-offs.** Haiku answers take about 41 seconds against Flash-Lite's 13, so the chat streams a live preview of the agent's thinking while it works. Only two models were re-judged by Gemini 3.1 Pro, to keep judge spend down.

---

## 3. How model quality is judged

**Context.** Cost and latency are easy to measure; answer quality is not. Decisions 1 and 2 rest on an LLM judge, so the judge had to be trustworthy.

**Decision.**
- **Blind and pairwise.** The judge sees two answers to the same question in random order, without knowing which system wrote which.
- **Evidence and date.** The judge also gets the tool outputs each agent saw and today's date, so it can check facts instead of relying on its own training data.
- **Unique factual errors.** Beyond picking a winner, the judge flags factual errors that only one answer makes.
- **A strong judge, then a judge from another model family.** Claude judges were checked against a Gemini judge for any result that favoured a Claude model.
- **Judge spend is tracked** alongside the runs it judges.

**Evidence.** The first judgement used Flash-Lite as judge with neither the date nor the evidence. It judged answers against a 2024 world and favoured the skills agent 13–8; with the date and evidence, stronger judges reversed that to 20–6. Under the Sonnet 5.5 judge, Flash-Lite's own answers carried 22 unique factual errors; under the Gemini 3.1 Pro judge, 4. Both judges ranked Haiku first, but the gap shows how much a judge's family shapes its scores.

**Trade-offs.** Strong judges cost real money: each Opus 5 judgement cost roughly $2 (an estimate from token counts; judge cost tracking was added afterwards), and the model comparison cost about $12 in runs and judging. The recommendation rule only counts the reference model as error-free when every one of its answers was judged.

---

## 4. Operator cost controls

**Context.** The hosted demo pays for LLM calls from anonymous visitors (the free trial) and signed-in users without their own keys.

**Decision.**
- **An allowlist.** The operator's key only runs models flagged `operator_lendable` in `agents/models.json`; any other model needs the user's own key, in chat, filings and briefings alike.
- **A per-call budget.** Each signed-in user has a daily budget, charged and enforced on every LLM call rather than per request, so a long question stops when the budget runs out instead of overshooting it. Calls are weighted by the model's price relative to the default model (a Gemini 3.6 Flash call counts as 8 Haiku 5.5 calls), so the budget means roughly the same dollar amount whichever allowed model runs.
- **A capped free trial.** Anonymous visitors get a few questions per IP per day on the free-trial model, under a global daily cap.

**Trade-offs.** A refused call can stop an answer midway; the user sees a clear "daily limit reached" message rather than a partial answer. The budget counts LLM calls, not tokens, so a long call costs the same as a short one.

---

## 5. A shared, persistent cache of filing analyses

**Context.** Summarising an SEC filing section is the most expensive step. Each server process used to keep its own in-memory cache, so analyses were lost on every restart and repeated for every user.

**Decision.** Store each analysis once per filing (ticker, form and accession number) in SQLite, shared by chat and the company page. Before serving one, check SEC EDGAR for a newer filing, at most every four hours per ticker. During an EDGAR outage, serve the last known filing instead of failing. Parse each filing once per accession and share the extraction between concurrent tools.

**Evidence.** Input tokens per filing-lookup question fell from 42,562 to 35,800 (−16%), and about 10% across all questions. A restarted server answers a repeated question without any analysis LLM calls (covered by tests, since each benchmark run starts from an empty database by design).

**Trade-offs.** A filing published inside the four-hour window can be missed until the window ends. Analyses are keyed by the filing actually read, so a refresh that lands mid-read can't save one filing's analysis under another's accession.

---

## 6. Identity and anonymous access

**Context.** The app should be usable without signing up, without letting anyone read or spend on another person's account.

**Decision.**
- The user id comes only from a verified Clerk session token; a client-supplied id must match it.
- Anonymous visitors can browse companies, charts and cached filing analyses, and use the free trial, but nothing they do is saved.
- Signing in unlocks saved chat history, a watchlist and briefings.
- Clerk's Frontend API is proxied through the app at `/__clerk`, because the site runs on a `vercel.app` domain where Clerk can't set DNS records.
- During a Clerk outage, tokens signed with keys the server already holds keep working. A token it can't check (an unknown key, or a misconfigured verifier) gets a 503 instead of silently becoming anonymous, and production refuses to start without `CLERK_ISSUER_URL`.

**Trade-offs.** Anonymous chats disappear when the tab closes. A long Clerk outage combined with a key rotation shows signed-in users an error rather than a degraded anonymous experience.

---

## 7. Tracing with OpenTelemetry and OpenInference

**Context.** Debugging a multi-step agent from logs alone was slow, and the traces needed to be readable by tools and agents, not only a vendor dashboard.

**Options.** LangSmith, or vendor-neutral OpenTelemetry with OpenInference's LangChain instrumentation.

**Decision.** OpenTelemetry with OpenInference, exported over OTLP to Railway's tracing. Tracing is off unless an OTLP endpoint is configured. User ids are hashed with a server-side secret before they reach a span, and API keys never do.

**Trade-offs.** When tracing is on, spans include full prompts and answers, which is what makes them useful for debugging. Keep it off in production unless that is acceptable.
