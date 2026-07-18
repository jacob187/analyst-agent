---
name: check-versions
description: Audit project dependencies against Context7 docs to find deprecated or incorrect API usage (model names, function signatures, config options). Use when the user says "check versions", "audit dependencies", "are we using the right models", or after hitting a NOT_FOUND / deprecated API error. Do NOT use for general research or feature planning.
---

# Check Versions

Scan the project for external library usage and verify correctness against current documentation via Context7.

## When This Skill Activates

The user wants to verify that the project's dependency usage (model IDs, API calls, config options) matches what's currently available. Common triggers:
- A 404 / NOT_FOUND error from an API (e.g., wrong model name)
- "Are we on the latest versions?"
- "Check our API usage is up to date"
- After upgrading a dependency

## Process

### 1. Identify Dependencies

Read the project's dependency file to get the library list:

```bash
# Python
cat pyproject.toml  # or requirements.txt, setup.py

# JavaScript/TypeScript
cat package.json
```

If the user specifies particular libraries, skip to step 2 for just those.

### 2. Extract Usage Patterns

For each library (or the ones the user cares about), search the codebase for how it's used:

- **Model IDs**: Grep for model name strings (e.g., `gemini-`, `gpt-`, `claude-`)
- **Import paths**: Grep for `from <library>` or `import <library>`
- **Config options**: Grep for constructor args, init params, config keys

Collect every unique usage into a checklist:

```
[ ] langchain-google-genai: model="gemini-3-flash-preview" (chat.py:72)
[ ] langchain-google-genai: model="gemini-2.5-flash" (watchlist.py:61)
[ ] yfinance: download(period="3mo") (get_stock_data.py:45)
```

### 3. Verify Against Context7

For each library with notable usage patterns:

1. Call `resolve-library-id` to get the Context7 library ID
2. Call `query-docs` with a targeted query about the specific usage (model names, API signatures, deprecated features)
3. Compare what the docs say vs. what the codebase uses

**Important**: Do NOT guess or rely on training data. The entire point is to use Context7 as the source of truth.

### 4. Report Findings

Present a table of results:

```markdown
## Version Audit Results

| Library | Usage | Location | Status | Action |
|---------|-------|----------|--------|--------|
| langchain-google-genai | model="gemini-3-flash-preview" | chat.py:72 | DEPRECATED | Update to `gemini-3.1-flash-preview` |
| yfinance | download(period="3mo") | get_stock_data.py:45 | OK | None |

### Recommended Changes
1. [File:line] — Change `X` to `Y` (reason)
2. ...
```

Categorize each finding:
- **OK** — Usage matches current docs
- **DEPRECATED** — Working now but flagged for removal
- **BROKEN** — Will fail or is already failing
- **UNKNOWN** — Context7 had no info; manual verification needed

### 5. Offer to Fix

```
Found [N] issues across [M] libraries.

Would you like me to:
1. Fix all issues automatically
2. Fix specific items (list numbers)
3. Save this report to docs/ for later
```

## Principles

- **Context7 is the source of truth.** Do not validate against training data — that's what caused the problem in the first place.
- **Be specific.** Report exact file paths, line numbers, and the current vs. recommended values.
- **Don't over-scope.** Only check libraries the project actually uses. Don't audit transitive dependencies unless asked.
- **Parallel lookups.** Use parallel subagents to resolve and query multiple libraries simultaneously when checking more than 2-3 libraries.
