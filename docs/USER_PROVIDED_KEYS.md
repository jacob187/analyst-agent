# Deploying with User-Provided API Keys

This guide explains how to deploy the Analyst Agent application where users provide their own API keys through the web interface, eliminating the need for server-side API key management.

## Overview

The application now supports two modes:

1. **User-Provided Keys** (Recommended for public deployment)
   - Users enter their own Google API Key and SEC header in the web UI
   - Keys are sent directly to the backend via WebSocket
   - No server-side API key storage required
   - Better for multi-tenant deployments

2. **Server-Side Keys** (Legacy mode)
   - Set `GOOGLE_API_KEY` and `SEC_HEADER` in `.env`
   - All users share the same API key
   - Simpler for single-user deployments

## Quick Start (User-Provided Keys)

### 1. Deploy Without API Keys

```bash
# Clone the repository
git clone https://github.com/yourusername/analyst-agent.git
cd analyst-agent

# Create minimal .env file (no API keys needed!)
cat > .env << EOF
PYTHONPATH="."
API_PORT=8000
FRONTEND_PORT=3000
EOF

# Start the services
docker-compose up --build
```

### 2. Access the Application

Open your browser to `http://localhost:3000`

You'll see the API key configuration screen:

```
┌─────────────────────────────────┐
│  Configure API Access           │
│  Your keys are only sent to     │
│  your self-hosted backend.      │
│  Never stored.                  │
├─────────────────────────────────┤
│  Google API Key                 │
│  [................................]  │
│                                 │
│  SEC User Agent                 │
│  [................................]  │
│                                 │
│  [x] Show keys                  │
│                   [Continue →]  │
└─────────────────────────────────┘
```

### 3. Users Enter Their Keys

**Google API Key:**
- Get one free at [Google AI Studio](https://aistudio.google.com/app/apikey)
- Free tier: 15 requests/minute, 1500 requests/day

**SEC User Agent:**
- Just enter your email address (required by SEC)
- Example: `yourname@example.com`

### 4. Start Analyzing

After entering keys:
1. Enter a ticker symbol (e.g., `AAPL`)
2. Ask questions about the company
3. Get AI-powered analysis of SEC filings

## Deployment Platforms

### Railway (Recommended)

Railway supports this deployment model perfectly:

```bash
# Install Railway CLI
npm install -g @railway/cli

# Login and initialize
railway login
railway init

# Deploy without API keys in environment
railway up

# Access your deployed app
railway open
```

### Vercel (Frontend) + Railway (Backend)

Split deployment for better scalability:

**Backend (Railway):**
```bash
cd analyst-agent
railway init
railway up
# Note your API URL: https://your-api.railway.app
```

**Frontend (Vercel):**
```bash
cd frontend

# Update src/lib/components/ChatWindow.svelte
# Change: ws://localhost:8000
# To: wss://your-api.railway.app

vercel --prod
```

### Docker Compose (Self-Hosted)

Perfect for VPS deployment (DigitalOcean, AWS EC2, etc.):

```bash
# On your server
git clone https://github.com/yourusername/analyst-agent.git
cd analyst-agent

# Create .env (no keys needed)
cp .env.example .env

# Start services
docker-compose up -d

# Setup nginx reverse proxy
sudo apt install nginx
```

**Nginx config** (`/etc/nginx/sites-available/analyst`):
```nginx
server {
    listen 80;
    server_name your-domain.com;

    location / {
        proxy_pass http://localhost:3000;
        proxy_http_version 1.1;
        proxy_set_header Upgrade $http_upgrade;
        proxy_set_header Connection 'upgrade';
        proxy_set_header Host $host;
    }

    location /ws/ {
        proxy_pass http://localhost:8000;
        proxy_http_version 1.1;
        proxy_set_header Upgrade $http_upgrade;
        proxy_set_header Connection "Upgrade";
        proxy_set_header Host $host;
    }

    location /health {
        proxy_pass http://localhost:8000;
    }
}
```

```bash
sudo ln -s /etc/nginx/sites-available/analyst /etc/nginx/sites-enabled/
sudo nginx -t
sudo systemctl restart nginx

# Optional: Setup SSL with Let's Encrypt
sudo apt install certbot python3-certbot-nginx
sudo certbot --nginx -d your-domain.com
```

## Security Considerations

### API Key Handling

**How it works:**
1. User enters API keys in browser
2. Keys are sent via WebSocket on connection
3. Backend validates keys by attempting to initialize the LLM
4. Keys are stored in memory for that WebSocket session only
5. Keys are discarded when connection closes

**Security benefits:**
- No API keys in environment variables
- No API keys in databases or files
- Each user uses their own quota
- Server compromise doesn't leak API keys

**Important notes:**
- Keys are sent over WebSocket (use WSS in production!)
- Keys are visible in browser memory (standard for web apps)
- Use HTTPS/WSS to prevent man-in-the-middle attacks

### Production Checklist

- [ ] Use HTTPS for frontend (free with Let's Encrypt/Vercel)
- [ ] Use WSS (secure WebSocket) for API connections
- [ ] Set up CORS properly in `api/main.py`
- [ ] Enable rate limiting on your reverse proxy
- [ ] Monitor server resources (CPU/memory)
- [ ] Set up logging and error tracking
- [ ] Add authentication if needed (OAuth, etc.)

## Cost Implications

### User-Provided Keys (This Setup)

**Server costs:**
- Railway: $5-10/month (small instance)
- DigitalOcean Droplet: $6/month (1GB RAM)
- AWS EC2 t3.micro: $7.50/month

**User costs (per user):**
- Google Gemini API: FREE tier
  - 15 requests/minute
  - 1500 requests/day
  - Sufficient for most analysis

**Total:** $5-10/month regardless of user count

### Server-Side Keys (Legacy)

**Costs:**
- Server: $5-10/month
- Google API: $0-1000+/month depending on all users' usage
- You pay for all API calls

**Problem:** One heavy user can rack up your bill

## Configuration Options

### Update Frontend WebSocket URL

For production deployment, update the WebSocket URL:

**File:** `frontend/src/lib/components/ChatWindow.svelte`

```typescript
// Development
const wsUrl = `ws://localhost:8000/ws/chat/${ticker}`;

// Production (use your domain)
const wsUrl = `wss://api.yourdomain.com/ws/chat/${ticker}`;

// Or use environment variable
const API_URL = import.meta.env.VITE_API_URL || 'http://localhost:8000';
const wsUrl = `${API_URL.replace('http', 'ws')}/ws/chat/${ticker}`;
```

Then set in `frontend/.env`:
```bash
VITE_API_URL=https://api.yourdomain.com
```

### Enable Server-Side Keys (Optional)

If you prefer server-side keys:

1. Update `.env`:
```bash
GOOGLE_API_KEY="AIza..."
SEC_HEADER="admin@yourcompany.com"
```

2. Update `api/main.py` to check for environment variables first:
```python
# In websocket handler, before waiting for auth message
google_api_key = os.getenv("GOOGLE_API_KEY")
sec_header = os.getenv("SEC_HEADER")

if google_api_key and sec_header:
    # Use server-side keys
    await websocket.send_json({
        "type": "auth_success",
        "message": f"Connected to {ticker}. Using server credentials."
    })
else:
    # Wait for user-provided keys (current behavior)
    auth_message = await websocket.receive_json()
    # ... rest of auth flow
```

3. Update frontend to skip API key input if server has keys

## FAQ

**Q: Are API keys stored anywhere?**
A: No. Keys exist only in memory during the WebSocket session and are discarded on disconnect.

**Q: Can I see what API calls users make?**
A: Yes, check the API logs: `docker-compose logs -f api`

**Q: What if a user enters invalid keys?**
A: The backend validates keys on connection. Invalid keys get an error message and the connection closes.

**Q: Can I limit which users can access the app?**
A: Yes, add authentication middleware. See `docs/AUTHENTICATION.md` (future)

**Q: Does this work with other LLM providers?**
A: Currently only Google Gemini. Support for OpenAI/Anthropic requires code changes.

**Q: How do I update the deployed app?**
A: Pull latest code and rebuild:
```bash
git pull origin main
docker-compose down
docker-compose up --build -d
```

## Troubleshooting

### WebSocket Connection Failed

**Symptom:** Frontend shows "Connection error"

**Fixes:**
1. Check API is running: `curl http://localhost:8000/health`
2. Check WebSocket URL matches your deployment
3. Check CORS settings in `api/main.py`
4. Check nginx WebSocket proxy config

### API Key Validation Failed

**Symptom:** "Failed to initialize agent" error

**Fixes:**
1. Verify Google API key is valid
2. Check API key has Gemini API enabled
3. Ensure no extra spaces in SEC header email
4. Check API logs: `docker-compose logs api`

### Frontend Build Failed

**Symptom:** Svelte build errors

**Fixes:**
```bash
cd frontend
rm -rf node_modules package-lock.json
npm install
npm run build
```

## Next Steps

- Add user authentication (OAuth, Auth0)
- Implement API usage tracking per user
- Add persistent chat history (PostgreSQL)
- Support multiple LLM providers
- Add streaming responses for real-time feel
- Deploy monitoring and analytics

## Support

- Issues: https://github.com/yourusername/analyst-agent/issues
- Docs: https://github.com/yourusername/analyst-agent/tree/main/docs
- Email: support@yourdomain.com
