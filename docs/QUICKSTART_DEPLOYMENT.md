# 🚀 Quick Start Deployment Guide

This guide will help you deploy the analyst-agent as a web application in under 2 hours.

---

## 🎯 Choose Your Path

### **Path 1: Railway (Recommended for MVP)** ⚡
- **Time:** 30 minutes
- **Cost:** $20/mo
- **Difficulty:** Easy
- **Best for:** Quick MVP, testing, demos

### **Path 2: Docker Compose (Self-hosted)**
- **Time:** 2 hours
- **Cost:** $10-20/mo (VPS)
- **Difficulty:** Medium
- **Best for:** Full control, development

### **Path 3: Kubernetes (Production)**
- **Time:** 1-2 weeks
- **Cost:** $500+/mo
- **Difficulty:** Hard
- **Best for:** Scale, enterprise

---

## ⚡ Path 1: Railway Deployment (30 minutes)

### Prerequisites
- GitHub account
- Credit card (for Railway)
- Domain name (optional)

### Step 1: Install Railway CLI
```bash
npm install -g @railway/cli
railway login
```

### Step 2: Initialize Project
```bash
cd analyst-agent
railway init
railway link  # Link to Railway project
```

### Step 3: Add Databases
```bash
# Add PostgreSQL
railway add postgresql

# Add Redis
railway add redis

# Set environment variables
railway variables set GOOGLE_API_KEY="your-api-key"
railway variables set SEC_HEADER="your-email@example.com Your App"
```

### Step 4: Create Backend (when ready)
```bash
cd backend
railway up
```

### Step 5: Deploy Frontend to Vercel
```bash
cd frontend
npm install -g vercel
vercel
```

### Step 6: Configure Domain
```bash
# In Railway dashboard
# Settings > Domains > Add custom domain
```

**Done! Your app is live at https://your-app.up.railway.app** 🎉

---

## 🐳 Path 2: Docker Compose (2 hours)

### Prerequisites
- Docker & Docker Compose installed
- Linux server (DigitalOcean, Linode, etc.)
- Domain name

### Step 1: Clone & Configure
```bash
git clone https://github.com/jacob187/analyst-agent.git
cd analyst-agent

# Copy environment file
cp .env.example .env

# Edit .env with your keys
nano .env
```

### Step 2: Start Services
```bash
# Start database & cache
docker-compose up -d db redis

# Check health
docker-compose ps

# View logs
docker-compose logs -f
```

### Step 3: Run Migrations (when backend is ready)
```bash
# Enter backend container
docker-compose exec api bash

# Run migrations
alembic upgrade head

# Create first user
python scripts/create_user.py
```

### Step 4: Configure Reverse Proxy (nginx)
```nginx
# /etc/nginx/sites-available/analyst-agent
server {
    listen 80;
    server_name your-domain.com;

    location / {
        proxy_pass http://localhost:3000;  # Frontend
        proxy_set_header Host $host;
        proxy_set_header X-Real-IP $remote_addr;
    }

    location /api {
        proxy_pass http://localhost:8000;  # Backend
        proxy_set_header Host $host;
        proxy_set_header X-Real-IP $remote_addr;
    }

    location /ws {
        proxy_pass http://localhost:8000;
        proxy_http_version 1.1;
        proxy_set_header Upgrade $http_upgrade;
        proxy_set_header Connection "upgrade";
    }
}
```

```bash
# Enable site
sudo ln -s /etc/nginx/sites-available/analyst-agent /etc/nginx/sites-enabled/
sudo nginx -t
sudo systemctl reload nginx

# Get SSL certificate
sudo certbot --nginx -d your-domain.com
```

### Step 5: Monitor
```bash
# View logs
docker-compose logs -f api worker

# Check resource usage
docker stats

# Restart services
docker-compose restart api
```

**Done! Your app is live at https://your-domain.com** 🎉

---

## 🏗️ Current Status & Next Steps

### ✅ Completed
- [x] Core analyst-agent logic (SEC + Yahoo Finance)
- [x] LangGraph chatbot
- [x] Test infrastructure
- [x] Docker Compose configuration
- [x] Deployment documentation

### 🚧 In Progress
- [ ] FastAPI backend (see roadmap)
- [ ] React frontend (see roadmap)
- [ ] Authentication system
- [ ] WebSocket chat

### 📋 To Build (Backend)

Create `backend/` directory with this structure:

```
backend/
├── app/
│   ├── main.py              # FastAPI app entry point
│   ├── config.py            # Settings (database URL, etc.)
│   ├── database.py          # SQLAlchemy setup
│   ├── models/              # Database models
│   │   ├── user.py
│   │   ├── ticker.py
│   │   └── analysis.py
│   ├── schemas/             # Pydantic schemas
│   │   ├── user.py
│   │   └── ticker.py
│   ├── api/                 # API routes
│   │   └── v1/
│   │       ├── endpoints/
│   │       │   ├── auth.py
│   │       │   ├── tickers.py
│   │       │   ├── sec.py
│   │       │   └── chat.py
│   │       └── api.py
│   ├── core/                # Core utilities
│   │   ├── auth.py          # JWT handling
│   │   └── security.py      # Password hashing
│   └── services/            # Business logic
│       ├── sec_service.py   # Wrapper around agents/sec_workflow
│       └── ai_service.py    # LangGraph integration
├── tests/
├── alembic/                 # Database migrations
├── Dockerfile
└── requirements.txt
```

**Quick Backend Setup:**
```bash
mkdir -p backend/app/{models,schemas,api/v1/endpoints,core,services}
cd backend

# Create requirements.txt
cat > requirements.txt << EOF
fastapi>=0.110.0
uvicorn[standard]>=0.27.0
sqlalchemy>=2.0.25
alembic>=1.13.1
psycopg2-binary>=2.9.9
redis>=5.0.1
celery>=5.3.6
pydantic>=2.6.0
pydantic-settings>=2.1.0
python-jose[cryptography]>=3.3.0
passlib[bcrypt]>=1.7.4
python-multipart>=0.0.9
slowapi>=0.1.9
pytest>=8.0.0
httpx>=0.26.0
EOF

# Install
pip install -r requirements.txt
```

### 📋 To Build (Frontend)

```bash
# Create React app
npm create vite@latest frontend -- --template react-ts
cd frontend

# Install dependencies
npm install @tanstack/react-query zustand
npm install socket.io-client axios
npm install react-router-dom
npm install @radix-ui/react-* # shadcn/ui components
npm install tailwindcss postcss autoprefixer
npm install recharts
npm install react-hook-form zod

# Initialize Tailwind
npx tailwindcss init -p
```

---

## 🔧 Development Workflow

### Local Development
```bash
# Terminal 1: Backend
cd backend
uvicorn app.main:app --reload

# Terminal 2: Frontend
cd frontend
npm run dev

# Terminal 3: Celery worker
cd backend
celery -A app.workers.celery_app worker -l info

# Terminal 4: Watch logs
docker-compose logs -f db redis
```

### Testing
```bash
# Backend tests
cd backend
pytest tests/ -v --cov

# Frontend tests
cd frontend
npm test
npm run test:e2e
```

### Database Migrations
```bash
# Create migration
alembic revision --autogenerate -m "Add users table"

# Apply migration
alembic upgrade head

# Rollback
alembic downgrade -1
```

---

## 📊 Monitoring

### Health Checks
```bash
# API health
curl http://localhost:8000/health

# Database
docker-compose exec db pg_isready

# Redis
docker-compose exec redis redis-cli ping
```

### Logs
```bash
# All services
docker-compose logs -f

# Specific service
docker-compose logs -f api

# Last 100 lines
docker-compose logs --tail=100 api
```

### Metrics (when Prometheus is set up)
- API: http://localhost:8000/metrics
- Grafana: http://localhost:3001

---

## 🆘 Troubleshooting

### Database Connection Issues
```bash
# Check database is running
docker-compose ps db

# Check connection
docker-compose exec db psql -U analyst -d analyst_agent

# Reset database
docker-compose down -v
docker-compose up -d db
```

### Redis Connection Issues
```bash
# Check Redis
docker-compose exec redis redis-cli ping

# Clear cache
docker-compose exec redis redis-cli FLUSHALL
```

### API Not Starting
```bash
# Check logs
docker-compose logs api

# Common issues:
# 1. Missing environment variables
# 2. Database not ready
# 3. Port already in use

# Fix port conflict
lsof -ti:8000 | xargs kill -9
```

### Frontend Not Loading
```bash
# Check API URL in .env
cat frontend/.env.local

# Should be:
VITE_API_URL=http://localhost:8000

# Rebuild
cd frontend
rm -rf node_modules dist
npm install
npm run dev
```

---

## 💡 Quick Tips

### Use ngrok for Testing
```bash
# Expose local backend to internet
ngrok http 8000

# Update frontend .env
VITE_API_URL=https://your-subdomain.ngrok.io
```

### Auto-reload Docker Compose
```yaml
# In docker-compose.yml
api:
  volumes:
    - ./backend:/app
  command: uvicorn app.main:app --reload --host 0.0.0.0
```

### Quick Database Backup
```bash
# Backup
docker-compose exec db pg_dump -U analyst analyst_agent > backup.sql

# Restore
docker-compose exec -T db psql -U analyst analyst_agent < backup.sql
```

---

## 📚 Next Steps

1. **Read the full deployment plan:** `docs/FULLSTACK_DEPLOYMENT_PLAN.md`
2. **Build the backend:** Follow FastAPI tutorial + integrate existing agents
3. **Build the frontend:** Follow React tutorial + design dashboard
4. **Deploy MVP:** Use Railway for quick testing
5. **Scale up:** Move to Kubernetes when you have users

---

## 🤝 Need Help?

- **Documentation:** `docs/` directory
- **Issues:** https://github.com/jacob187/analyst-agent/issues
- **Architecture:** See `docs/FULLSTACK_DEPLOYMENT_PLAN.md`
- **MCP Integration:** See `docs/MCP_INTEGRATION_PLAN.md`

---

## 📈 Success Metrics

Track these to know when you're ready for production:

- [ ] API responds in < 200ms (95th percentile)
- [ ] Frontend loads in < 2 seconds
- [ ] Chat messages stream smoothly
- [ ] 99.9% uptime for 1 week
- [ ] All tests passing
- [ ] Security audit completed
- [ ] Load tested (100+ concurrent users)
- [ ] Backup & recovery tested

**Good luck! 🚀**
