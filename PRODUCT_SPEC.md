# Stock Analyst Agent - Product Specification

## Overview

Transform the existing AI stock analysis prototype into a production-ready SaaS application positioned as a "light Bloomberg terminal" for retail investors.

**Target Price:** $5/mo base tier, $15-30/mo premium tier
**Timeline:** 8 weeks (4-5 hour weekly sprints)
**Total Development Time:** ~36 hours

---

## Current State

### Existing Features
- SEC filing analysis (10-K, 10-Q) with sentiment scoring
- Technical indicators via Yahoo Finance (RSI, MACD, Bollinger Bands)
- WebSocket-based AI chat interface
- LangGraph ReAct agent architecture
- Basic Svelte frontend with dark/light theme
- Docker deployment setup

### Gaps for Production
- No persistent data storage
- No user-specific features (watchlists, history)
- No structured data views (dashboards, tables)
- No proactive features (alerts, notifications)
- No export capabilities
- Missing landing/marketing pages

---

## Target State

A polished SaaS application where users can:
1. Track stocks via personalized watchlists
2. View structured dashboards with key metrics
3. Chat with AI about any stock with full conversation history
4. Receive alerts on price movements and SEC filings
5. Access insider trading data and earnings calendars
6. Export professional PDF reports
7. (Premium) Run deep research queries via Tavily

---

## Technical Decisions

### Database Schema (Prepared for Auth Integration)

```sql
-- Users table (skeleton - auth developer will expand)
CREATE TABLE users (
  id UUID PRIMARY KEY,
  email TEXT UNIQUE,
  tier TEXT DEFAULT 'free', -- 'free', 'basic', 'premium'
  created_at TIMESTAMP DEFAULT NOW()
);

-- Watchlist
CREATE TABLE watchlist (
  id UUID PRIMARY KEY DEFAULT gen_random_uuid(),
  user_id UUID REFERENCES users(id),
  ticker TEXT NOT NULL,
  added_at TIMESTAMP DEFAULT NOW(),
  notes TEXT,
  UNIQUE(user_id, ticker)
);

-- Chat sessions
CREATE TABLE chat_sessions (
  id UUID PRIMARY KEY DEFAULT gen_random_uuid(),
  user_id UUID REFERENCES users(id),
  ticker TEXT NOT NULL,
  title TEXT,
  created_at TIMESTAMP DEFAULT NOW(),
  updated_at TIMESTAMP DEFAULT NOW()
);

-- Chat messages
CREATE TABLE chat_messages (
  id UUID PRIMARY KEY DEFAULT gen_random_uuid(),
  session_id UUID REFERENCES chat_sessions(id) ON DELETE CASCADE,
  role TEXT NOT NULL, -- 'user' or 'assistant'
  content TEXT NOT NULL,
  created_at TIMESTAMP DEFAULT NOW()
);

-- Price alerts
CREATE TABLE price_alerts (
  id UUID PRIMARY KEY DEFAULT gen_random_uuid(),
  user_id UUID REFERENCES users(id),
  ticker TEXT NOT NULL,
  condition TEXT NOT NULL, -- 'above' or 'below'
  target_price DECIMAL NOT NULL,
  triggered BOOLEAN DEFAULT FALSE,
  created_at TIMESTAMP DEFAULT NOW()
);

-- SEC filing alerts (tracks last seen filing)
CREATE TABLE sec_filing_alerts (
  id UUID PRIMARY KEY DEFAULT gen_random_uuid(),
  user_id UUID REFERENCES users(id),
  ticker TEXT NOT NULL,
  last_filing_date DATE,
  UNIQUE(user_id, ticker)
);

-- Deep research usage (for premium tier tracking)
CREATE TABLE deep_research_usage (
  id UUID PRIMARY KEY DEFAULT gen_random_uuid(),
  user_id UUID REFERENCES users(id),
  query TEXT NOT NULL,
  credits_used INTEGER,
  created_at TIMESTAMP DEFAULT NOW()
);
```

### API Endpoints (New)

```
# Watchlist
GET    /api/watchlist              - Get user's watchlist
POST   /api/watchlist              - Add ticker to watchlist
DELETE /api/watchlist/{ticker}     - Remove from watchlist
PATCH  /api/watchlist/{ticker}     - Update notes

# Chat History
GET    /api/sessions               - List user's chat sessions
GET    /api/sessions/{id}          - Get session with messages
POST   /api/sessions               - Create new session
DELETE /api/sessions/{id}          - Delete session

# Stock Data
GET    /api/stock/{ticker}/summary - Key metrics dashboard data
GET    /api/stock/{ticker}/insider - Insider transactions
GET    /api/stock/{ticker}/earnings - Earnings history & calendar
GET    /api/stock/{ticker}/chart   - OHLCV data for charts

# Alerts
GET    /api/alerts                 - Get user's alerts
POST   /api/alerts/price           - Create price alert
POST   /api/alerts/filing          - Create SEC filing alert
DELETE /api/alerts/{id}            - Delete alert

# Export
POST   /api/export/pdf             - Generate PDF report

# Premium
POST   /api/research/deep          - Tavily deep research (tier-gated)
```

### Frontend Pages (New)

```
/                    - Landing page (marketing)
/app                 - Main app shell (authenticated)
/app/dashboard       - Watchlist overview with metrics
/app/stock/{ticker}  - Stock detail view with tabs
/app/chat/{ticker}   - Chat interface (existing, enhanced)
/app/alerts          - Manage alerts
/app/history         - Past chat sessions
/app/settings        - User preferences
```

---

## Sprint Plan

### Week 1: Database & API Foundation
**Goal:** Set up persistent storage and core API structure

**Tasks:**
- [ ] Set up PostgreSQL database (local + Supabase project)
- [ ] Create database schema (tables above)
- [ ] Build base CRUD API endpoints for watchlist
- [ ] Add SQLAlchemy/asyncpg to backend
- [ ] Test database connections

**Deliverable:** Working watchlist API (add/remove/list tickers)

**Files to Create/Modify:**
- `api/database.py` - Database connection
- `api/models.py` - SQLAlchemy models
- `api/routers/watchlist.py` - Watchlist endpoints
- `api/main.py` - Register new routers

---

### Week 2: Chat Persistence
**Goal:** Save and restore chat conversations

**Tasks:**
- [ ] Build chat sessions API endpoints
- [ ] Modify WebSocket handler to save messages to DB
- [ ] Add session management (create, list, load, delete)
- [ ] Implement agent memory (pass full history to LLM)
- [ ] Test conversation continuity

**Deliverable:** Chat history persists across sessions

**Files to Create/Modify:**
- `api/routers/sessions.py` - Session endpoints
- `api/main.py` - Update WebSocket handler
- `agents/graph/sec_graph.py` - Add memory to agent

---

### Week 3: Stock Dashboard API
**Goal:** Structured data endpoints for dashboard views

**Tasks:**
- [ ] Build `/api/stock/{ticker}/summary` endpoint
  - Current price, change, volume
  - P/E, market cap, 52-week range
  - Key financials (revenue, EPS, margins)
- [ ] Build `/api/stock/{ticker}/insider` endpoint
  - Recent Form 4 filings
  - Buy/sell transactions with amounts
- [ ] Build `/api/stock/{ticker}/earnings` endpoint
  - Historical EPS beats/misses
  - Next earnings date
- [ ] Add caching layer for API responses

**Deliverable:** All dashboard data available via API

**Files to Create/Modify:**
- `api/routers/stock.py` - Stock data endpoints
- `api/services/stock_data.py` - Data fetching logic
- `api/services/insider_data.py` - SEC Form 4 parsing

---

### Week 4: Frontend - Dashboard & Watchlist UI
**Goal:** Build the main dashboard interface

**Tasks:**
- [ ] Create app layout shell with navigation
- [ ] Build watchlist component (add/remove stocks)
- [ ] Build stock card component (mini metrics view)
- [ ] Create dashboard page showing watchlist with cards
- [ ] Add loading states and error handling

**Deliverable:** Functional dashboard showing watchlist with metrics

**Files to Create/Modify:**
- `frontend/src/routes/app/+layout.svelte` - App shell
- `frontend/src/routes/app/dashboard/+page.svelte` - Dashboard
- `frontend/src/lib/components/WatchlistCard.svelte`
- `frontend/src/lib/components/AddStockModal.svelte`
- `frontend/src/lib/api.ts` - API client functions

---

### Week 5: Frontend - Stock Detail Page
**Goal:** Comprehensive single-stock view

**Tasks:**
- [ ] Create stock detail page with tabs
- [ ] Integrate TradingView widget for charts
- [ ] Build metrics panel (key financials grid)
- [ ] Build insider transactions table
- [ ] Build earnings history component
- [ ] Connect existing chat to new layout

**Deliverable:** Full stock detail page with charts, metrics, insider data

**Files to Create/Modify:**
- `frontend/src/routes/app/stock/[ticker]/+page.svelte`
- `frontend/src/lib/components/TradingViewChart.svelte`
- `frontend/src/lib/components/MetricsPanel.svelte`
- `frontend/src/lib/components/InsiderTable.svelte`
- `frontend/src/lib/components/EarningsHistory.svelte`

---

### Week 6: Alerts System
**Goal:** Price and SEC filing alerts

**Tasks:**
- [ ] Build alerts API endpoints
- [ ] Create price monitoring background job
- [ ] Create SEC filing check background job
- [ ] Build alerts management UI
- [ ] Add alert creation modals on stock page
- [ ] Implement in-app notifications

**Deliverable:** Working alerts system (in-app notifications)

**Files to Create/Modify:**
- `api/routers/alerts.py` - Alerts endpoints
- `api/jobs/price_monitor.py` - Price check job
- `api/jobs/sec_monitor.py` - SEC filing check job
- `frontend/src/routes/app/alerts/+page.svelte`
- `frontend/src/lib/components/AlertModal.svelte`
- `frontend/src/lib/components/NotificationBell.svelte`

---

### Week 7: Export & Polish
**Goal:** PDF export and UI refinement

**Tasks:**
- [ ] Build PDF report generation endpoint
- [ ] Design report template (metrics, chart snapshot, AI summary)
- [ ] Add export buttons throughout UI
- [ ] Create landing page with value proposition
- [ ] Add pricing section to landing page
- [ ] Mobile responsiveness pass
- [ ] Loading skeletons and empty states

**Deliverable:** Export functionality, polished landing page

**Files to Create/Modify:**
- `api/routers/export.py` - PDF generation
- `api/services/pdf_generator.py` - Report template
- `frontend/src/routes/+page.svelte` - Landing page
- Various components - Polish pass

---

### Week 8: Premium Features & Integration Prep
**Goal:** Deep research feature, prepare for auth integration

**Tasks:**
- [ ] Integrate Tavily deep research API
- [ ] Build deep research UI (separate from chat)
- [ ] Add usage tracking for premium features
- [ ] Create tier-gating middleware (placeholder for auth)
- [ ] Document all API endpoints for auth developer
- [ ] Create environment variable template
- [ ] Test full user flow end-to-end
- [ ] Bug fixes and final polish

**Deliverable:** Complete feature set, ready for auth integration

**Files to Create/Modify:**
- `api/routers/research.py` - Deep research endpoint
- `api/services/tavily_client.py` - Tavily integration
- `api/middleware/tier_gate.py` - Tier checking (stub)
- `frontend/src/routes/app/research/+page.svelte`
- `AUTH_INTEGRATION.md` - Handoff doc for auth developer

---

## Feature Specifications

### Watchlist

**Behavior:**
- Users can add any valid ticker to watchlist
- Maximum 50 stocks per user (free tier), unlimited (premium)
- Each watchlist item can have user notes
- Watchlist displays mini cards with: price, daily change, sparkline

**UI:**
- Dashboard shows grid of watchlist cards
- "Add Stock" button opens modal with ticker search
- Click card to go to stock detail page
- Swipe/button to remove from watchlist

---

### Chat History

**Behavior:**
- Each conversation is tied to a ticker
- Conversations persist indefinitely
- Agent receives full conversation history for context
- Users can delete conversations
- Auto-title conversations based on first message

**UI:**
- Sidebar shows list of past conversations grouped by date
- Click to load conversation
- "New Chat" button starts fresh conversation
- Delete button with confirmation

---

### Stock Dashboard

**Metrics Displayed:**
```
Price & Change     | P/E Ratio        | Market Cap
52-Week High/Low   | Volume (vs avg)  | Dividend Yield
EPS (TTM)          | Revenue (TTM)    | Profit Margin
Beta               | Avg Analyst Rating| Next Earnings
```

**Insider Transactions Table:**
```
Date | Insider Name | Title | Type (Buy/Sell) | Shares | Price | Value
```

**Earnings History:**
- Chart showing EPS estimates vs actual (last 8 quarters)
- Beat/miss indicator
- Next earnings date with countdown

---

### TradingView Integration

**Implementation:**
```html
<!-- TradingView Widget -->
<div class="tradingview-widget-container">
  <div id="tradingview_{ticker}"></div>
</div>

<script src="https://s3.tradingview.com/tv.js"></script>
<script>
  new TradingView.widget({
    "width": "100%",
    "height": 500,
    "symbol": "{ticker}",
    "interval": "D",
    "theme": "dark",
    "style": "1",
    "locale": "en",
    "toolbar_bg": "#f1f3f6",
    "enable_publishing": false,
    "allow_symbol_change": false,
    "studies": ["RSI@tv-basicstudies", "MACD@tv-basicstudies"]
  });
</script>
```

---

### Price Alerts

**Behavior:**
- Set alert when price goes above or below target
- Check prices every 5 minutes via background job
- Mark alert as triggered (one-time, doesn't repeat)
- Maximum 10 alerts (free), 50 alerts (premium)

**Alert Check Job (Pseudocode):**
```python
async def check_price_alerts():
    active_alerts = await db.get_active_alerts()

    # Group by ticker to minimize API calls
    tickers = set(a.ticker for a in active_alerts)
    prices = await get_current_prices(tickers)

    for alert in active_alerts:
        current = prices[alert.ticker]
        if alert.condition == 'above' and current >= alert.target_price:
            await trigger_alert(alert)
        elif alert.condition == 'below' and current <= alert.target_price:
            await trigger_alert(alert)
```

---

### SEC Filing Alerts

**Behavior:**
- Monitor for new 10-K, 10-Q, 8-K filings
- Check daily via background job
- Notify when new filing detected
- Store last seen filing date per ticker

---

### PDF Export

**Report Contents:**
1. Header with ticker, company name, date generated
2. Key metrics table
3. Price chart snapshot (last 6 months)
4. AI-generated executive summary
5. Recent insider transactions
6. Risk factors summary (from latest 10-K/10-Q)
7. Disclaimer footer

**Implementation:**
- Use `weasyprint` or `reportlab` for PDF generation
- Generate chart image server-side with `matplotlib`
- Cache generated reports for 1 hour

---

### Tavily Deep Research (Premium)

**Behavior:**
- Separate from regular chat
- User enters research query
- System runs Tavily deep research
- Display comprehensive report with sources
- Track usage: 5/month (basic premium), 20/month (top tier)

**UI:**
- Dedicated "Research" page
- Large text input for research query
- "Run Deep Research" button
- Loading state (can take 30-60 seconds)
- Results displayed as formatted report with citations
- Usage counter shown

---

## Landing Page Content

### Hero Section
**Headline:** "SEC Filings + AI Analysis. No Bloomberg Terminal Required."

**Subhead:** "Ask questions about any stock. Get answers backed by actual SEC data, not news summaries."

**CTA:** "Start Analyzing - $5/month"

### Feature Highlights
1. **Direct SEC Access** - "We read the actual 10-K, not articles about it"
2. **Real-Time Data** - "Live prices and technical indicators"
3. **Insider Tracking** - "See what executives are buying and selling"
4. **AI That Cites Sources** - "Every answer backed by filings"

### Pricing Section
```
BASIC - $5/mo
- Unlimited AI chat
- Watchlist (50 stocks)
- Price alerts (10)
- SEC filing alerts
- PDF exports

ANALYST - $15/mo
- Everything in Basic
- Unlimited watchlist
- Price alerts (50)
- Deep research (5/month)
- Priority support

PRO - $30/mo
- Everything in Analyst
- Deep research (20/month)
- API access
- Custom reports
```

---

## Auth Integration Handoff

When auth developer joins, provide:

1. **Database Schema** - Tables above, they add auth fields to users
2. **API Middleware Pattern** - Where to inject user context
3. **Tier Gating Points** - Which endpoints need tier checks
4. **Environment Variables** - Supabase keys, Stripe keys needed
5. **User Flow** - Sign up → Stripe checkout → Access granted

**Files They'll Modify:**
- `api/middleware/auth.py` - Create auth middleware
- `api/dependencies.py` - User injection for routes
- `api/routers/*.py` - Add `current_user` dependency
- `frontend/src/lib/auth.ts` - Supabase client
- `frontend/src/routes/login/+page.svelte` - Login page
- `frontend/src/routes/signup/+page.svelte` - Signup flow

---

## Environment Variables

```bash
# Database
DATABASE_URL=postgresql://user:pass@host:5432/analyst_agent

# APIs (existing)
GOOGLE_API_KEY=your_gemini_key
SEC_HEADER=your_email@example.com

# APIs (new)
TAVILY_API_KEY=your_tavily_key

# Auth (for auth developer)
SUPABASE_URL=https://xxx.supabase.co
SUPABASE_ANON_KEY=xxx
SUPABASE_SERVICE_KEY=xxx
STRIPE_SECRET_KEY=sk_xxx
STRIPE_WEBHOOK_SECRET=whsec_xxx
```

---

## Success Criteria

**End of Week 8, the product should:**
- [ ] Display personalized watchlist with live metrics
- [ ] Show comprehensive stock detail pages with charts
- [ ] Persist chat conversations across sessions
- [ ] Send in-app notifications for price/filing alerts
- [ ] Generate professional PDF reports
- [ ] Run Tavily deep research queries
- [ ] Have polished landing page ready for launch
- [ ] Be fully documented for auth integration

**Ready for auth developer to:**
- [ ] Add Supabase authentication
- [ ] Connect Stripe subscriptions
- [ ] Implement tier-based access control
- [ ] Deploy to production

---

## Risk Mitigation

| Risk | Mitigation |
|------|------------|
| Yahoo Finance rate limits | Add caching, batch requests |
| SEC API downtime | Cache filings, graceful degradation |
| Tavily costs spike | Hard caps per user, usage tracking |
| Scope creep | Strict sprint goals, defer nice-to-haves |
| TradingView widget issues | Fallback to simple chart.js charts |

---

## Post-Launch Roadmap (Future)

- Portfolio tracking with P&L
- Email notifications (not just in-app)
- Stock screener with custom filters
- Comparison tool (side-by-side stocks)
- Mobile app (PWA or React Native)
- API access for premium users
- Webhook integrations
