# SaaS Transformation Plan — Stock Analyst Agent

## Overview

This document outlines the implementation plan for transforming the Stock Analyst Agent from a single-user tool into a multi-tenant SaaS application. The plan is ordered by dependency — each phase builds on the previous one.

---

## Phase 1: User Authentication & Multi-Tenancy (Foundation)

**Goal**: Multiple users can sign up, log in, and have isolated data.

### 1.1 Database Migration (SQLite → PostgreSQL)

- Replace `aiosqlite` with `asyncpg` + SQLAlchemy async ORM
- Add Alembic for schema migrations
- Why now: adding `user_id` foreign keys to existing tables is a natural migration point, and SQLite won't scale for concurrent multi-user access

**New `users` table:**

```sql
CREATE TABLE users (
    id UUID PRIMARY KEY DEFAULT gen_random_uuid(),
    email TEXT UNIQUE NOT NULL,
    password_hash TEXT NOT NULL,
    display_name TEXT,
    plan TEXT DEFAULT 'free',  -- free | pro | enterprise
    created_at TIMESTAMP DEFAULT NOW(),
    updated_at TIMESTAMP DEFAULT NOW()
);
```

**Modified existing tables:**

```sql
-- sessions: add user ownership
ALTER TABLE sessions ADD COLUMN user_id UUID REFERENCES users(id) ON DELETE CASCADE;
CREATE INDEX idx_sessions_user ON sessions(user_id);

-- messages: inherits isolation via session foreign key (no change needed)

-- settings: becomes per-user
ALTER TABLE settings ADD COLUMN user_id UUID UNIQUE REFERENCES users(id) ON DELETE CASCADE;
-- Remove the id=1 single-row constraint
```

### 1.2 Auth System

- **Library**: `python-jose` for JWT + `passlib[bcrypt]` for password hashing
- **Flow**: email+password registration → JWT access token (15 min) + refresh token (7 days)
- **Storage**: refresh tokens in a `refresh_tokens` table with expiry and revocation support

**New API endpoints:**

| Method | Path | Description |
|--------|------|-------------|
| POST | `/auth/register` | Create account |
| POST | `/auth/login` | Get JWT tokens |
| POST | `/auth/refresh` | Refresh access token |
| POST | `/auth/logout` | Revoke refresh token |
| GET | `/auth/me` | Get current user profile |

**WebSocket auth change:**
- Current: `auth` message sends raw API keys
- New: `auth` message sends JWT token; server validates and loads user's stored API keys from DB

**FastAPI middleware:**

```python
from fastapi import Depends
from fastapi.security import HTTPBearer

async def get_current_user(token: str = Depends(HTTPBearer())) -> User:
    payload = jwt.decode(token, SECRET_KEY, algorithms=["HS256"])
    user = await db.get_user(payload["sub"])
    if not user:
        raise HTTPException(401)
    return user
```

### 1.3 Per-User API Key Encryption

- Encrypt API keys at rest using `cryptography.fernet` with a server-side key
- Each user stores their own Gemini, SEC, and Tavily keys
- Keys are decrypted only when creating an agent session

### 1.4 Frontend Auth Flow

- Add login/register pages in Svelte
- Store JWT in memory (not localStorage) with refresh token in httpOnly cookie
- Add auth guard to chat routes
- Show user profile / logout in header

**Estimated scope**: ~15-20 new files, modifications to `api/main.py`, `api/db.py`, and frontend components.

---

## Phase 2: Usage Tracking & Rate Limiting

**Goal**: Track per-user usage and enforce limits by plan tier.

### 2.1 Usage Tracking Table

```sql
CREATE TABLE usage_events (
    id UUID PRIMARY KEY DEFAULT gen_random_uuid(),
    user_id UUID REFERENCES users(id),
    event_type TEXT NOT NULL,  -- 'query' | 'tool_call' | 'llm_tokens'
    metadata JSONB,            -- {tool_name, token_count, ticker, etc.}
    created_at TIMESTAMP DEFAULT NOW()
);
CREATE INDEX idx_usage_user_date ON usage_events(user_id, created_at);
```

### 2.2 Plan Tiers & Limits

| Feature | Free | Pro ($29/mo) | Enterprise ($99/mo) |
|---------|------|-------------|---------------------|
| Queries/day | 20 | 500 | Unlimited |
| Chat sessions | 5 | Unlimited | Unlimited |
| SEC filings tools | Yes | Yes | Yes |
| Web research tools | No | Yes | Yes |
| Chat history retention | 7 days | 90 days | Unlimited |
| Priority support | No | No | Yes |

### 2.3 Rate Limiting Middleware

- Use `slowapi` or custom middleware backed by Redis
- Check user's plan tier → enforce daily/hourly limits
- Return `429 Too Many Requests` with `Retry-After` header when exceeded
- WebSocket: send `rate_limited` message type instead of disconnecting

---

## Phase 3: Billing & Subscriptions (Stripe)

**Goal**: Users can subscribe to paid plans and manage billing.

### 3.1 Stripe Integration

- `stripe` Python SDK for server-side
- Stripe Checkout for payment flow (no need to handle card details)
- Stripe Webhooks for subscription lifecycle events

**New endpoints:**

| Method | Path | Description |
|--------|------|-------------|
| POST | `/billing/create-checkout` | Start Stripe Checkout session |
| POST | `/billing/webhook` | Handle Stripe events |
| GET | `/billing/subscription` | Get current subscription status |
| POST | `/billing/portal` | Redirect to Stripe Customer Portal |

**Webhook events to handle:**
- `checkout.session.completed` → upgrade user plan
- `customer.subscription.updated` → plan change
- `customer.subscription.deleted` → downgrade to free
- `invoice.payment_failed` → notify user, grace period

### 3.2 Database Additions

```sql
ALTER TABLE users ADD COLUMN stripe_customer_id TEXT UNIQUE;
ALTER TABLE users ADD COLUMN subscription_status TEXT DEFAULT 'none';
-- none | active | past_due | canceled
```

---

## Phase 4: Platform-Managed API Keys (Optional BYOK → Managed)

**Goal**: Users don't need their own API keys — the platform provides LLM access.

### 4.1 Approach

- Platform holds master Gemini/Tavily API keys
- Pro/Enterprise users get access without bringing their own keys
- Free users still use BYOK (or get limited platform-managed access)
- LLM token usage is metered and counted against plan limits

### 4.2 Cost Management

- Track token usage per user via `usage_events`
- Set per-user daily token budgets based on plan tier
- Alert users approaching limits via WebSocket status messages

---

## Phase 5: Operational Features

### 5.1 Admin Dashboard

- Separate admin frontend (or admin section behind role check)
- Metrics: active users, queries/day, revenue, popular tickers
- User management: view/edit/suspend accounts
- System health: API latency, error rates

### 5.2 Audit Logging

```sql
CREATE TABLE audit_log (
    id UUID PRIMARY KEY DEFAULT gen_random_uuid(),
    user_id UUID REFERENCES users(id),
    action TEXT NOT NULL,
    resource_type TEXT,
    resource_id TEXT,
    metadata JSONB,
    ip_address INET,
    created_at TIMESTAMP DEFAULT NOW()
);
```

### 5.3 Team/Workspace Support (Enterprise)

- Workspaces with shared sessions and analysis
- Role-based access: owner, admin, analyst, viewer
- Shared API key pool per workspace

---

## Implementation Priority

```
Phase 1 (Auth + Multi-tenancy)     ← START HERE
    ↓
Phase 2 (Usage Tracking + Limits)
    ↓
Phase 3 (Stripe Billing)
    ↓
Phase 4 (Managed API Keys)
    ↓
Phase 5 (Admin + Teams)
```

Phase 1 is the critical path — everything else depends on having users. Phases 2 and 3 can be partially parallelized. Phases 4 and 5 are enhancements on a working SaaS.

---

## Key Technical Decisions

| Decision | Recommendation | Rationale |
|----------|---------------|-----------|
| Database | PostgreSQL (via Supabase or managed) | Multi-tenant concurrency, JSONB, full-text search |
| Auth | Custom JWT (not a third-party auth service) | Full control, simpler for this scale |
| Billing | Stripe Checkout + Customer Portal | Minimal PCI scope, handles subscriptions |
| Caching/Rate Limiting | Redis | Fast counters, TTL support |
| Deployment | Docker Compose → managed containers | Easy to start, easy to scale |
| Secret Management | Environment variables + Fernet encryption | Simple, secure enough for early stage |
