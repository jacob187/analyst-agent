# MCP Server Integration Plan

## Overview
Based on Issue #1, we're evaluating the integration of [mcp-yfinance-ux](https://github.com/bxxd/mcp-yfinance-ux) MCP server to replace or complement our custom Yahoo Finance wrapper.

## MCP Server Analysis

### Capabilities
The `mcp-yfinance-ux` server provides three main tools:

1. **markets()** - Complete market landscape view
   - US equities overview
   - Global indices
   - All 11 GICS sectors
   - Commodity prices
   - Volatility metrics (VIX, etc.)
   - Interest rates

2. **sector(name)** - Detailed sector analysis
   - Sector ETF data
   - Momentum indicators
   - Top 10 holdings with betas
   - Relative strength metrics

3. **ticker(symbol)** - Individual security analysis
   - Factor exposures (beta, volatility)
   - Valuation metrics (P/E, P/B, etc.)
   - Calendar events (earnings, dividends)
   - Momentum indicators (RSI, Moving Averages)
   - 52-week ranges
   - Supports batch comparison

### Key Features
- ✅ Bloomberg Terminal-style formatting (dense, scannable tables)
- ✅ Optimized for AI agent consumption
- ✅ Parallel data fetching with ThreadPoolExecutor
- ✅ Batch operations for multiple tickers
- ✅ Professional-grade calculations (Beta, idiosyncratic volatility)

### Critical Limitations ⚠️
The MCP server has **important restrictions** per the project documentation:

> "yfinance is an unofficial web scraper, not a sanctioned Yahoo Finance API"

**Restrictions:**
- ❌ **NOT suitable for automation** (cron jobs, background monitoring)
- ❌ **NOT suitable for production infrastructure**
- ❌ **NOT suitable for real-time tracking**
- ⚠️ Vulnerable to Yahoo blocking and site changes
- ⚠️ Designed for "ad-hoc, user-initiated requests only"

**Recommendation from developers:**
For production use, use official APIs:
- Alpha Vantage
- Polygon.io
- IEX Cloud

## Recommendation: Hybrid Approach

Given the limitations, I recommend a **hybrid architecture** rather than complete replacement:

### Architecture Design

```
┌─────────────────────────────────────────────────────────┐
│                    Analyst Agent                        │
├─────────────────────────────────────────────────────────┤
│                                                          │
│  ┌──────────────────┐        ┌────────────────────┐   │
│  │  LangGraph Agent │        │  Main Workflows    │   │
│  │  (User Queries)  │        │  (Automated Data)  │   │
│  └────────┬─────────┘        └─────────┬──────────┘   │
│           │                             │               │
│           ├─────────────┬───────────────┤              │
│           │             │               │               │
│  ┌────────▼─────┐  ┌───▼────────┐ ┌────▼──────────┐  │
│  │ MCP yfinance │  │ SEC Tools  │ │ yfinance      │  │
│  │ Tools (3)    │  │ (8 tools)  │ │ Wrapper       │  │
│  │              │  │            │ │ (Background)  │  │
│  │ - markets()  │  │            │ │               │  │
│  │ - sector()   │  │            │ │ - Historical  │  │
│  │ - ticker()   │  │            │ │ - Financials  │  │
│  └──────────────┘  └────────────┘ │ - Indicators  │  │
│                                    └───────────────┘  │
└─────────────────────────────────────────────────────────┘
```

### Use Cases by Component

#### **1. Custom yfinance Wrapper** (Keep for automation)
**Use for:**
- ✅ Automated data fetching (scheduled updates)
- ✅ Background technical indicator calculations
- ✅ Historical data retrieval for analysis
- ✅ Building datasets for reports
- ✅ Batch processing multiple tickers
- ✅ Integration with main_agent.py workflows

**Implementation:** Current `YahooFinanceDataRetrieval` class

#### **2. MCP yfinance-ux Server** (Add for user queries)
**Use for:**
- ✅ Interactive user-initiated queries via chatbot
- ✅ Ad-hoc market exploration
- ✅ Comparative analysis (multiple tickers)
- ✅ Sector-level insights
- ✅ Market overview requests
- ✅ LangGraph agent tool selection

**Implementation:** New MCP client integration as LangGraph tools

## Implementation Plan

### Phase 1: Add MCP Tools to LangGraph Agent
1. Install MCP client dependencies
2. Create MCP tool wrappers in `agents/tools/yfinance_mcp_tools.py`
3. Integrate with existing LangGraph agent
4. Add to `create_sec_agent()` tool list

### Phase 2: Keep Existing Wrapper for Automation
1. Maintain `agents/technical_workflow/get_stock_data.py`
2. Use for `main_agent.py` batch workflows
3. Use for background indicator calculations
4. Use for report generation

### Phase 3: Testing
1. Unit tests for MCP tool wrappers
2. Integration tests for LangGraph agent with MCP tools
3. E2E tests for user query workflows

### Phase 4: Documentation
1. Update README with MCP server setup instructions
2. Document when to use MCP vs custom wrapper
3. Add examples of MCP tool usage

## Benefits of Hybrid Approach

✅ **Flexibility**: Use MCP for interactive queries, wrapper for automation
✅ **Reliability**: Don't depend on MCP server for critical workflows
✅ **Best of Both Worlds**: Bloomberg-style formatting + automation capabilities
✅ **Gradual Migration**: Can test MCP integration without breaking existing functionality
✅ **User Experience**: Better formatting for chatbot responses
✅ **Production Ready**: Automated workflows don't violate MCP usage guidelines

## Alternative: Full Migration (Not Recommended)

If we were to fully replace the custom wrapper:

❌ **Risks:**
- Violate "no automation" guideline
- Vulnerable to Yahoo blocking
- Cannot run scheduled updates
- Cannot use for production dashboards
- Breaks existing main_agent.py workflows

✅ **Benefits:**
- Less code to maintain
- Consistent data source

**Verdict:** Not worth the trade-offs

## Implementation Code Structure

```
agents/
├── tools/
│   ├── sec_tools.py (existing)
│   ├── yfinance_mcp_tools.py (NEW - MCP integration)
│   └── __init__.py
├── technical_workflow/
│   ├── get_stock_data.py (KEEP - for automation)
│   ├── process_technical_indicators.py (KEEP)
│   └── main_technical_workflow.py (KEEP)
└── graph/
    ├── sec_graph.py (UPDATE - add MCP tools)
    └── combined_graph.py (NEW - SEC + Technical + MCP)
```

## Next Steps

1. ✅ Document test infrastructure (DONE)
2. ✅ Research MCP server (DONE)
3. ⏭️ Implement MCP tool wrappers
4. ⏭️ Add MCP tools to LangGraph agent
5. ⏭️ Write tests for MCP integration
6. ⏭️ Update CLI to expose MCP capabilities
7. ⏭️ Document usage patterns

## Conclusion

**Recommended Action**: Implement hybrid approach
- Add MCP tools for interactive user queries
- Keep custom wrapper for automated workflows
- Best aligns with both project goals and MCP server limitations
