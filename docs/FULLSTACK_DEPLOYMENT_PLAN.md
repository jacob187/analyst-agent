# Full-Stack Deployment Plan: Analyst Agent Platform

## 🎯 Vision: AI-Powered Financial Analysis Platform
Transform analyst-agent into a production SaaS application - think "Bloomberg Terminal meets ChatGPT"

---

## 📐 Architecture Overview

```
┌─────────────────────────────────────────────────────────────────┐
│                         USERS (Web Browser)                     │
└───────────────────────────┬─────────────────────────────────────┘
                            │ HTTPS
┌───────────────────────────▼─────────────────────────────────────┐
│                     FRONTEND (React + TypeScript)               │
│  ┌──────────────┐  ┌──────────────┐  ┌─────────────────────┐  │
│  │  Dashboard   │  │  Chat UI     │  │  Charts & Reports   │  │
│  │  Component   │  │  Component   │  │  Component          │  │
│  └──────────────┘  └──────────────┘  └─────────────────────┘  │
│                                                                  │
│  Deployment: Vercel/Netlify/CloudFlare Pages                   │
└───────────────────────────┬─────────────────────────────────────┘
                            │ WebSocket + REST/GraphQL
┌───────────────────────────▼─────────────────────────────────────┐
│                     API GATEWAY / LOAD BALANCER                 │
│                     (nginx/Caddy/CloudFlare)                    │
└───────────────────────────┬─────────────────────────────────────┘
                            │
┌───────────────────────────▼─────────────────────────────────────┐
│                   BACKEND (FastAPI + Python)                    │
│  ┌──────────────────────────────────────────────────────────┐  │
│  │  REST API Endpoints                                       │  │
│  │  - /api/v1/tickers/{ticker}                              │  │
│  │  - /api/v1/analysis/sec/{ticker}                         │  │
│  │  - /api/v1/analysis/technical/{ticker}                   │  │
│  │  - /api/v1/chat (WebSocket)                              │  │
│  │  - /api/v1/export/{format}                               │  │
│  └──────────────────────────────────────────────────────────┘  │
│                                                                  │
│  ┌──────────────────────────────────────────────────────────┐  │
│  │  Background Workers (Celery)                             │  │
│  │  - Data refresh jobs                                      │  │
│  │  - Report generation                                      │  │
│  │  - Email notifications                                    │  │
│  └──────────────────────────────────────────────────────────┘  │
│                                                                  │
│  Deployment: Docker + Kubernetes / Railway / Render            │
└───────────────┬──────────────────┬──────────────────────────────┘
                │                  │
     ┌──────────▼──────┐    ┌──────▼──────────────┐
     │   PostgreSQL    │    │   Redis             │
     │   (User Data,   │    │   (Cache, Sessions, │
     │    Analysis     │    │    Job Queue)       │
     │    Results)     │    └─────────────────────┘
     └─────────────────┘
                │
     ┌──────────▼──────────────────────────────────┐
     │   External APIs                             │
     │   - SEC EDGAR                               │
     │   - Yahoo Finance (yfinance)                │
     │   - Google Gemini (LLM)                     │
     │   - MCP Server (optional)                   │
     └─────────────────────────────────────────────┘
```

---

## 🛠️ Technology Stack

### **Frontend**
```typescript
Framework:      React 18+ with TypeScript
Build Tool:     Vite (fast dev server, optimized builds)
Styling:        Tailwind CSS + shadcn/ui components
State Mgmt:     Zustand (lightweight) or Redux Toolkit
Data Fetching:  TanStack Query (React Query)
Charts:         TradingView Lightweight Charts / Recharts
WebSocket:      Socket.io-client
Forms:          React Hook Form + Zod validation
Testing:        Vitest + React Testing Library + Playwright
```

### **Backend**
```python
Framework:      FastAPI (async, high performance)
ORM:            SQLAlchemy 2.0 + Alembic (migrations)
Task Queue:     Celery + Redis
WebSocket:      FastAPI WebSocket + Socket.io
Auth:           JWT tokens + OAuth2 (Google, GitHub)
API Docs:       Auto-generated OpenAPI (Swagger)
Testing:        pytest + pytest-asyncio + httpx
Monitoring:     Sentry (error tracking) + Prometheus
```

### **Database & Cache**
```
Primary DB:     PostgreSQL 15+ (Supabase or managed)
Cache:          Redis 7+ (Upstash or managed)
File Storage:   S3-compatible (AWS S3, Cloudflare R2)
Vector DB:      Pinecone/Weaviate (optional, for embeddings)
```

### **DevOps & Infrastructure**
```
Containers:     Docker + Docker Compose
Orchestration:  Kubernetes (GKE/EKS) or Railway/Render
CI/CD:          GitHub Actions
Monitoring:     Grafana + Prometheus + Loki
Secrets:        Vault or Doppler
CDN:            CloudFlare
```

---

## 📅 Implementation Roadmap (12 Weeks)

### **Phase 1: Backend API Foundation** (Weeks 1-3)

#### Week 1: Project Setup & Core API
```bash
# Create backend structure
analyst-agent/
├── backend/
│   ├── app/
│   │   ├── __init__.py
│   │   ├── main.py              # FastAPI app
│   │   ├── config.py            # Settings (Pydantic)
│   │   ├── database.py          # DB connection
│   │   ├── models/              # SQLAlchemy models
│   │   ├── schemas/             # Pydantic schemas
│   │   ├── api/
│   │   │   ├── v1/
│   │   │   │   ├── endpoints/
│   │   │   │   │   ├── auth.py
│   │   │   │   │   ├── tickers.py
│   │   │   │   │   ├── sec.py
│   │   │   │   │   ├── technical.py
│   │   │   │   │   ├── chat.py
│   │   │   │   │   └── export.py
│   │   │   │   └── api.py       # Router
│   │   ├── core/
│   │   │   ├── auth.py          # JWT, OAuth
│   │   │   ├── deps.py          # Dependencies
│   │   │   └── security.py      # Password hashing
│   │   ├── services/            # Business logic
│   │   │   ├── sec_service.py
│   │   │   ├── yahoo_service.py
│   │   │   └── ai_service.py
│   │   └── workers/
│   │       └── tasks.py         # Celery tasks
│   ├── tests/
│   ├── alembic/                 # DB migrations
│   ├── Dockerfile
│   └── requirements.txt
├── frontend/                     # (Week 4+)
└── docker-compose.yml
```

**Key Tasks:**
- [ ] Set up FastAPI project with proper structure
- [ ] Implement PostgreSQL models (User, Analysis, Ticker, etc.)
- [ ] Create authentication (JWT + OAuth2)
- [ ] Build REST endpoints for ticker data
- [ ] Integrate existing analyst-agent code
- [ ] Add rate limiting (slowapi)
- [ ] Set up Redis caching
- [ ] Write API tests (pytest + httpx)

#### Week 2: LangGraph Integration & WebSocket
```python
# backend/app/api/v1/endpoints/chat.py
from fastapi import WebSocket
from langgraph.checkpoint.aiosqlite import AsyncSqliteSaver
from agents.graph.sec_graph import create_sec_agent

@router.websocket("/ws/chat/{ticker}")
async def chat_endpoint(websocket: WebSocket, ticker: str):
    await websocket.accept()

    # Create agent with persistent memory
    agent = create_sec_agent(llm, ticker)

    async for message in websocket.iter_text():
        # Stream LLM responses token-by-token
        async for chunk in agent.astream_events({"messages": [message]}):
            await websocket.send_json(chunk)
```

**Key Tasks:**
- [ ] Implement WebSocket chat endpoint
- [ ] Add streaming LLM responses
- [ ] Integrate LangGraph agents
- [ ] Add conversation persistence to PostgreSQL
- [ ] Implement rate limiting per user
- [ ] Add error handling & reconnection logic

#### Week 3: Background Jobs & Data Pipeline
```python
# backend/app/workers/tasks.py
from celery import Celery
from agents.main_agent import MainAgent

celery = Celery(__name__, broker="redis://localhost:6379")

@celery.task
def refresh_ticker_data(ticker: str):
    """Background job to refresh ticker data"""
    agent = MainAgent(ticker)
    agent.run()
    # Save to PostgreSQL

@celery.task
def generate_report(user_id: int, ticker: str, format: str):
    """Generate PDF/Excel report"""
    # Use existing markdown generation + conversion
```

**Key Tasks:**
- [ ] Set up Celery + Redis
- [ ] Create background jobs for data refresh
- [ ] Implement report generation (PDF, Excel)
- [ ] Add job status tracking
- [ ] Set up scheduled tasks (cron-like)

---

### **Phase 2: Frontend Development** (Weeks 4-7)

#### Week 4: Project Setup & Authentication
```bash
# Create frontend
cd frontend
npm create vite@latest . -- --template react-ts
npm install @tanstack/react-query zustand
npm install tailwindcss @shadcn/ui
npm install recharts socket.io-client
npm install react-hook-form zod
```

**Key Components:**
```typescript
// src/components/auth/LoginPage.tsx
import { useAuth } from '@/hooks/useAuth'

export function LoginPage() {
  // OAuth login with Google/GitHub
  // JWT token management
  // Protected routes
}
```

**Key Tasks:**
- [ ] Set up React + TypeScript + Vite
- [ ] Configure Tailwind CSS + shadcn/ui
- [ ] Implement authentication flow
- [ ] Create protected route wrapper
- [ ] Set up API client (axios/fetch)
- [ ] Configure TanStack Query

#### Week 5: Dashboard & Data Visualization
```typescript
// src/components/dashboard/Dashboard.tsx
import { TradingViewChart } from '@/components/charts/TradingViewChart'
import { TickerSearch } from '@/components/search/TickerSearch'
import { MetricsPanel } from '@/components/metrics/MetricsPanel'

export function Dashboard() {
  return (
    <div className="grid grid-cols-12 gap-4">
      <div className="col-span-8">
        <TradingViewChart ticker={selectedTicker} />
      </div>
      <div className="col-span-4">
        <MetricsPanel ticker={selectedTicker} />
      </div>
    </div>
  )
}
```

**Key Tasks:**
- [ ] Build ticker search component
- [ ] Integrate TradingView charts
- [ ] Create metrics dashboard
- [ ] Add technical indicators panel
- [ ] Implement responsive layout
- [ ] Add loading states & skeletons

#### Week 6: Chat Interface
```typescript
// src/components/chat/ChatInterface.tsx
import { useWebSocket } from '@/hooks/useWebSocket'

export function ChatInterface({ ticker }: { ticker: string }) {
  const { messages, sendMessage, isConnected } = useWebSocket(ticker)

  return (
    <div className="chat-container">
      <MessageList messages={messages} />
      <ChatInput onSend={sendMessage} disabled={!isConnected} />
    </div>
  )
}
```

**Key Tasks:**
- [ ] Build chat UI component
- [ ] Implement WebSocket connection
- [ ] Add message streaming display
- [ ] Support markdown rendering
- [ ] Add conversation history
- [ ] Implement typing indicators

#### Week 7: Reports & Export
```typescript
// src/components/export/ExportButton.tsx
export function ExportButton({ ticker }: { ticker: string }) {
  const exportToPDF = async () => {
    const response = await fetch(`/api/v1/export/pdf/${ticker}`)
    const blob = await response.blob()
    downloadFile(blob, `${ticker}_analysis.pdf`)
  }

  return (
    <DropdownMenu>
      <DropdownMenuItem onClick={exportToPDF}>Export PDF</DropdownMenuItem>
      <DropdownMenuItem onClick={exportToExcel}>Export Excel</DropdownMenuItem>
      <DropdownMenuItem onClick={exportToCSV}>Export CSV</DropdownMenuItem>
    </DropdownMenu>
  )
}
```

**Key Tasks:**
- [ ] Build export UI components
- [ ] Implement PDF generation
- [ ] Implement Excel export
- [ ] Add CSV export
- [ ] Create email report feature

---

### **Phase 3: Testing & Polish** (Weeks 8-9)

#### Week 8: Testing
- [ ] Write frontend unit tests (Vitest)
- [ ] Write frontend component tests (React Testing Library)
- [ ] Write E2E tests (Playwright)
- [ ] Backend API tests (100+ tests)
- [ ] Integration tests
- [ ] Load testing (Locust/K6)

#### Week 9: Performance & UX
- [ ] Optimize bundle size (code splitting)
- [ ] Add service worker (PWA)
- [ ] Implement lazy loading
- [ ] Add error boundaries
- [ ] Improve loading states
- [ ] Add tooltips & help text
- [ ] Accessibility audit (WCAG 2.1)

---

### **Phase 4: Deployment** (Weeks 10-12)

#### Week 10: Infrastructure Setup

**Option 1: Simple (Railway/Render) - $20-50/mo**
```yaml
# railway.toml or render.yaml
services:
  - type: web
    name: api
    env: docker
    dockerfilePath: ./backend/Dockerfile
    envVars:
      - key: DATABASE_URL
        fromDatabase: postgresql
      - key: REDIS_URL
        fromDatabase: redis

  - type: postgresql
    name: db

  - type: redis
    name: cache
```

**Option 2: Production (AWS/GCP/Azure) - $100-500/mo**
```yaml
# kubernetes/deployment.yaml
apiVersion: apps/v1
kind: Deployment
metadata:
  name: analyst-agent-api
spec:
  replicas: 3
  template:
    spec:
      containers:
      - name: api
        image: your-registry/analyst-agent:latest
        resources:
          requests:
            memory: "512Mi"
            cpu: "500m"
```

**Key Tasks:**
- [ ] Set up Docker images
- [ ] Configure CI/CD (GitHub Actions)
- [ ] Set up managed PostgreSQL
- [ ] Set up managed Redis
- [ ] Configure domain & SSL
- [ ] Set up CDN (CloudFlare)
- [ ] Configure environment variables

#### Week 11: Monitoring & Security
```python
# backend/app/main.py
from prometheus_fastapi_instrumentator import Instrumentator
import sentry_sdk

# Error tracking
sentry_sdk.init(dsn="your-sentry-dsn")

# Metrics
Instrumentator().instrument(app).expose(app)

# Security headers
@app.middleware("http")
async def add_security_headers(request, call_next):
    response = await call_next(request)
    response.headers["X-Frame-Options"] = "DENY"
    response.headers["X-Content-Type-Options"] = "nosniff"
    response.headers["Strict-Transport-Security"] = "max-age=31536000"
    return response
```

**Key Tasks:**
- [ ] Set up Sentry error tracking
- [ ] Configure Prometheus metrics
- [ ] Set up Grafana dashboards
- [ ] Add security headers
- [ ] Implement rate limiting
- [ ] Set up backup strategy
- [ ] Configure log aggregation

#### Week 12: Beta Launch
- [ ] Create landing page
- [ ] Set up user onboarding
- [ ] Add analytics (PostHog/Mixpanel)
- [ ] Create documentation
- [ ] Set up feedback system
- [ ] Invite beta users
- [ ] Monitor & fix issues

---

## 💰 Cost Breakdown (Monthly)

### **Minimal Setup (Hobby/MVP)** - $30-60/mo
```
Railway/Render:          $20 (backend + DB + Redis)
Vercel/Netlify:          $0 (frontend - free tier)
Domain:                  $1/mo (Namecheap)
Sentry:                  $0 (free tier)
External APIs:           $10-30 (LLM usage)
───────────────────────────────
Total:                   $31-51/mo
```

### **Small Scale (100-500 users)** - $150-300/mo
```
Railway/Render Pro:      $50-100
Supabase Pro:           $25 (PostgreSQL)
Upstash:                $10 (Redis)
Vercel Pro:             $20 (frontend)
CloudFlare:             $0-20 (CDN)
Sentry Team:            $26
Google Gemini API:      $20-100 (usage-based)
───────────────────────────────
Total:                   $151-301/mo
```

### **Production Scale (1K-10K users)** - $500-2000/mo
```
AWS/GCP/Azure:
  - EKS/GKE:            $150 (cluster)
  - EC2/Compute:        $200 (3x t3.medium)
  - RDS PostgreSQL:     $100 (db.t3.medium)
  - ElastiCache:        $50 (Redis)
  - S3/Storage:         $20
  - Load Balancer:      $20
  - CloudWatch:         $30

Vercel Pro:             $20
CloudFlare Pro:         $20
Sentry Business:        $80
Google Gemini:          $200-500
Email (SendGrid):       $15
───────────────────────────────
Total:                   $905-1205/mo
```

---

## 🔐 Security Considerations

### **Critical Security Measures**

1. **Authentication & Authorization**
```python
# Implement proper JWT with refresh tokens
# Use OAuth2 for social login
# Add 2FA for sensitive operations
# Implement role-based access control (RBAC)
```

2. **Data Protection**
```python
# Encrypt sensitive data at rest (PII, API keys)
# Use HTTPS everywhere (TLS 1.3)
# Implement proper CORS policies
# Sanitize user inputs (prevent XSS, SQL injection)
```

3. **API Security**
```python
# Rate limiting (per user, per IP)
# API key rotation
# Input validation (Pydantic)
# CSRF protection for state-changing operations
```

4. **LLM Security**
```python
# Prompt injection prevention
# Content filtering
# Token usage limits per user
# Audit logs for LLM interactions
```

5. **Compliance**
- **GDPR**: User data deletion, data export
- **SOC 2**: Security controls, audit logs
- **Financial Data**: Disclaimer that data is for informational purposes only

---

## ⚖️ Legal Considerations

### **Data Usage Restrictions**

⚠️ **Yahoo Finance (yfinance)**
```
CRITICAL: yfinance is an unofficial scraper
- NOT approved by Yahoo
- Terms of Service may prohibit commercial use
- Can be blocked at any time
- NOT suitable for production SaaS

RECOMMENDATION: Use official APIs for production
```

### **Alternatives for Production**

1. **Free Tier Options**
   - Alpha Vantage (500 calls/day free)
   - IEX Cloud (50K messages/mo free)
   - Finnhub (60 calls/min free)

2. **Paid Options**
   - Polygon.io ($29-199/mo)
   - Alpha Vantage Premium ($50-500/mo)
   - Bloomberg API (enterprise pricing)

3. **SEC Data** ✅
   - SEC EDGAR is free and official
   - Requires proper User-Agent header
   - Rate limit: ~10 requests/second

### **Content Licensing**
```
✅ Your code: MIT license (keep it)
⚠️ LLM outputs: Check Gemini's commercial use policy
⚠️ Financial data: Add disclaimer
⚠️ Charts/visualizations: TradingView free/paid license
```

### **Required Disclaimers**
```
"This platform is for informational and educational purposes only.
Not financial advice. Not endorsed by any financial institution.
Data may be inaccurate or delayed. Consult a licensed professional."
```

---

## 📈 Monetization Strategy

### **Pricing Tiers**

```
FREE TIER ($0/mo)
- 10 ticker analyses per month
- Basic chat (5 messages/day)
- Public data only
- Community support

PRO TIER ($29/mo)
- Unlimited ticker analyses
- Unlimited chat
- Real-time data
- Export to PDF/Excel
- Email reports
- Priority support

TEAM TIER ($99/mo)
- Everything in Pro
- Up to 5 users
- Shared watchlists
- API access
- Custom alerts
- Slack/Discord integration

ENTERPRISE (Custom)
- White-label deployment
- Dedicated infrastructure
- SLA guarantees
- Custom integrations
- Premium support
```

---

## 🚀 Quick Start Deployment (Minimal Viable Product)

### **Option 1: Railway (Easiest)** ⚡

```bash
# 1. Install Railway CLI
npm install -g @railway/cli

# 2. Login
railway login

# 3. Create project
railway init

# 4. Add PostgreSQL & Redis
railway add postgresql redis

# 5. Deploy backend
cd backend
railway up

# 6. Deploy frontend on Vercel
cd ../frontend
npm install -g vercel
vercel
```

**Estimated time: 2 hours**
**Cost: $20/mo**

### **Option 2: Docker Compose (Self-hosted)** 🐳

```yaml
# docker-compose.yml
version: '3.8'
services:
  api:
    build: ./backend
    ports:
      - "8000:8000"
    environment:
      - DATABASE_URL=postgresql://user:pass@db:5432/analyst
      - REDIS_URL=redis://redis:6379
    depends_on:
      - db
      - redis

  frontend:
    build: ./frontend
    ports:
      - "3000:3000"

  db:
    image: postgres:15
    volumes:
      - postgres_data:/var/lib/postgresql/data

  redis:
    image: redis:7

  celery:
    build: ./backend
    command: celery -A app.workers worker -l info

volumes:
  postgres_data:
```

```bash
# Deploy
docker-compose up -d
```

**Estimated time: 4 hours**
**Cost: $10-20/mo (VPS)**

---

## 📊 Scalability Plan

### **Traffic Estimates**

| Users | Requests/Day | DB Size | Infrastructure |
|-------|--------------|---------|----------------|
| 100   | 10K          | 1 GB    | Railway/Render |
| 1K    | 100K         | 10 GB   | Railway Pro    |
| 10K   | 1M           | 100 GB  | Kubernetes     |
| 100K  | 10M          | 1 TB    | Multi-region   |

### **Bottlenecks & Solutions**

1. **LLM API Rate Limits**
   - Solution: Queue requests, implement caching
   - Cache LLM responses for 24 hours
   - Use smaller models for simple queries

2. **Database Queries**
   - Solution: Add read replicas
   - Implement query caching (Redis)
   - Use materialized views

3. **SEC/Yahoo API Limits**
   - Solution: Cache aggressively
   - Pre-fetch popular tickers
   - Use CDN for static data

---

## ✅ Next Steps (Action Items)

### **This Week**
1. [ ] Decide on deployment strategy (Railway vs K8s)
2. [ ] Choose data provider (yfinance limitations vs paid APIs)
3. [ ] Set up GitHub repository structure (mono-repo vs multi-repo)
4. [ ] Create project board with milestones

### **This Month**
1. [ ] Build FastAPI backend skeleton
2. [ ] Set up PostgreSQL + migrations
3. [ ] Create first API endpoints
4. [ ] Set up React frontend skeleton
5. [ ] Deploy MVP to Railway

### **Next 3 Months**
1. [ ] Complete full backend implementation
2. [ ] Complete full frontend implementation
3. [ ] Write comprehensive tests
4. [ ] Deploy to production
5. [ ] Launch beta program

---

## 🎓 Learning Resources

### **Backend (FastAPI)**
- [FastAPI Official Tutorial](https://fastapi.tiangolo.com/tutorial/)
- [SQLAlchemy 2.0 Tutorial](https://docs.sqlalchemy.org/en/20/tutorial/)
- [Celery Best Practices](https://docs.celeryproject.org/en/stable/userguide/tasks.html)

### **Frontend (React)**
- [React Official Docs](https://react.dev/)
- [TanStack Query](https://tanstack.com/query/latest)
- [shadcn/ui Components](https://ui.shadcn.com/)

### **Deployment**
- [Railway Docs](https://docs.railway.app/)
- [Kubernetes Tutorial](https://kubernetes.io/docs/tutorials/)
- [Docker Best Practices](https://docs.docker.com/develop/dev-best-practices/)

---

## 📝 Summary

To turn analyst-agent into a production SaaS:

1. **Build FastAPI backend** - Weeks 1-3
2. **Build React frontend** - Weeks 4-7
3. **Test & polish** - Weeks 8-9
4. **Deploy & monitor** - Weeks 10-12

**Minimal MVP**: Railway + Vercel (~$30/mo, 2 hours setup)
**Production**: Kubernetes + PostgreSQL (~$500/mo, 12 weeks)

**Critical Decision**: Replace yfinance with official API for production use

**Estimated Total Time**: 300-400 hours (3-4 months full-time)
**Estimated MVP Cost**: $30-60/mo
**Estimated Production Cost**: $500-2000/mo
