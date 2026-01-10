# 🚀 Weekend Sprint: Ship in 48 Hours

Transform analyst-agent into a live web app this weekend. No frills, just shipping.

---

## 🎯 Weekend Goal

**By Sunday 8pm, you'll have:**
- Live web app with chat interface
- AI analysis for any ticker (using your existing LangGraph agent)
- Deployed and shareable URL
- Working demo for user feedback

**What we're SKIPPING for speed:**
- ❌ Authentication (add Monday)
- ❌ Database (use file storage)
- ❌ Background jobs (run synchronously)
- ❌ Beautiful UI (MVP only)
- ❌ Tests (add next week)

---

## 📅 48-Hour Timeline

### **Saturday Morning (9am-12pm): Backend**
**Goal: FastAPI wrapping your existing code**

### **Saturday Afternoon (1pm-6pm): Frontend**
**Goal: React app with chat + display**

### **Saturday Evening (7pm-10pm): Integration**
**Goal: Connect frontend to backend**

### **Sunday Morning (9am-12pm): Deploy**
**Goal: Live on Railway**

### **Sunday Afternoon (1pm-6pm): Polish & Test**
**Goal: Fix bugs, make it usable**

### **Sunday Evening (7pm-8pm): Launch**
**Goal: Share with first users**

---

## 🛠️ Saturday Morning: Minimal Backend (3 hours)

### Step 1: Create Backend Structure (15 min)

```bash
cd analyst-agent
mkdir -p backend/app
cd backend

# Create files
touch app/__init__.py
touch app/main.py
touch app/config.py
touch requirements.txt
touch Dockerfile
```

### Step 2: Install Dependencies (10 min)

```bash
# backend/requirements.txt
fastapi==0.110.0
uvicorn[standard]==0.27.0
python-dotenv==1.1.0
pydantic==2.6.0
pydantic-settings==2.1.0

# Copy your existing dependencies
edgartools>=3.14.2
langchain>=0.3.22
langchain-google-genai>=2.1.4
langgraph>=0.6.4
yfinance>=0.2.55

# Install
pip install -r requirements.txt
```

### Step 3: Create Minimal FastAPI App (1 hour)

```python
# backend/app/config.py
from pydantic_settings import BaseSettings

class Settings(BaseSettings):
    GOOGLE_API_KEY: str
    SEC_HEADER: str
    CORS_ORIGINS: str = "http://localhost:3000,http://localhost:5173"

    class Config:
        env_file = "../.env"

settings = Settings()
```

```python
# backend/app/main.py
from fastapi import FastAPI, WebSocket
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
import sys
import os

# Add parent directory to path to import agents
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "../.."))

from agents.graph.sec_graph import create_sec_agent
from langchain_google_genai import ChatGoogleGenerativeAI
from app.config import settings

app = FastAPI(title="Analyst Agent API")

# CORS
app.add_middleware(
    CORSMiddleware,
    allow_origins=settings.CORS_ORIGINS.split(","),
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Simple in-memory storage (replace with DB later)
agent_cache = {}

@app.get("/")
def root():
    return {"message": "Analyst Agent API", "version": "0.1.0"}

@app.get("/health")
def health():
    return {"status": "healthy"}

class AnalysisRequest(BaseModel):
    ticker: str

@app.post("/api/v1/analyze")
async def analyze(request: AnalysisRequest):
    """Quick analysis endpoint (synchronous for MVP)"""
    try:
        from agents.main_agent import MainAgent

        agent = MainAgent(request.ticker.upper())
        # This runs synchronously - good enough for MVP
        result = agent.run()

        return {
            "ticker": request.ticker.upper(),
            "status": "success",
            "data": result
        }
    except Exception as e:
        return {
            "ticker": request.ticker,
            "status": "error",
            "error": str(e)
        }

@app.websocket("/ws/chat/{ticker}")
async def chat_endpoint(websocket: WebSocket, ticker: str):
    """WebSocket for chat (simplified)"""
    await websocket.accept()

    try:
        # Create or get agent
        if ticker not in agent_cache:
            llm = ChatGoogleGenerativeAI(
                model="gemini-2.0-flash-exp",
                google_api_key=settings.GOOGLE_API_KEY
            )
            agent, _ = create_sec_agent(llm, ticker.upper())
            agent_cache[ticker] = agent

        agent = agent_cache[ticker]

        # Chat loop
        while True:
            message = await websocket.receive_text()

            # Invoke agent (synchronous for MVP)
            response = agent.invoke({"messages": [("user", message)]})

            # Extract last message
            if "messages" in response and response["messages"]:
                last_message = response["messages"][-1]
                reply = last_message.content if hasattr(last_message, 'content') else str(last_message)
            else:
                reply = str(response)

            await websocket.send_json({
                "type": "message",
                "content": reply
            })

    except Exception as e:
        await websocket.send_json({
            "type": "error",
            "content": str(e)
        })
        await websocket.close()

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)
```

### Step 4: Test Backend (15 min)

```bash
# Terminal 1: Start backend
cd backend
python -m app.main

# Terminal 2: Test endpoints
curl http://localhost:8000/
curl http://localhost:8000/health
curl -X POST http://localhost:8000/api/v1/analyze \
  -H "Content-Type: application/json" \
  -d '{"ticker": "AAPL"}'
```

**✅ Saturday Morning Checkpoint:** Backend running on localhost:8000

---

## 🎨 Saturday Afternoon: Minimal Frontend (5 hours)

### Step 1: Create React App (20 min)

```bash
cd analyst-agent
npm create vite@latest frontend -- --template react-ts
cd frontend

# Install dependencies
npm install
npm install socket.io-client
npm install @tanstack/react-query
npm install axios

# Start dev server (test it works)
npm run dev
```

### Step 2: Create Chat Component (2 hours)

```bash
# Create components
mkdir -p src/components
touch src/components/ChatInterface.tsx
touch src/components/TickerInput.tsx
```

```typescript
// src/components/ChatInterface.tsx
import { useState, useEffect, useRef } from 'react'
import { io, Socket } from 'socket.io-client'

interface Message {
  role: 'user' | 'assistant'
  content: string
}

export function ChatInterface({ ticker }: { ticker: string }) {
  const [messages, setMessages] = useState<Message[]>([])
  const [input, setInput] = useState('')
  const [socket, setSocket] = useState<WebSocket | null>(null)
  const [connected, setConnected] = useState(false)
  const messagesEndRef = useRef<HTMLDivElement>(null)

  useEffect(() => {
    // Connect to WebSocket
    const ws = new WebSocket(`ws://localhost:8000/ws/chat/${ticker}`)

    ws.onopen = () => {
      setConnected(true)
      console.log('Connected to chat')
    }

    ws.onmessage = (event) => {
      const data = JSON.parse(event.data)
      if (data.type === 'message') {
        setMessages(prev => [...prev, {
          role: 'assistant',
          content: data.content
        }])
      }
    }

    ws.onerror = (error) => {
      console.error('WebSocket error:', error)
    }

    ws.onclose = () => {
      setConnected(false)
      console.log('Disconnected from chat')
    }

    setSocket(ws)

    return () => ws.close()
  }, [ticker])

  const sendMessage = () => {
    if (!input.trim() || !socket || !connected) return

    // Add user message
    setMessages(prev => [...prev, {
      role: 'user',
      content: input
    }])

    // Send to server
    socket.send(input)
    setInput('')
  }

  useEffect(() => {
    messagesEndRef.current?.scrollIntoView({ behavior: 'smooth' })
  }, [messages])

  return (
    <div style={{
      display: 'flex',
      flexDirection: 'column',
      height: '600px',
      border: '1px solid #ccc',
      borderRadius: '8px',
      padding: '16px'
    }}>
      <h2>Chat with AI Analyst - {ticker}</h2>

      <div style={{
        flex: 1,
        overflowY: 'auto',
        marginBottom: '16px',
        padding: '8px'
      }}>
        {messages.map((msg, i) => (
          <div key={i} style={{
            marginBottom: '12px',
            padding: '12px',
            borderRadius: '8px',
            backgroundColor: msg.role === 'user' ? '#e3f2fd' : '#f5f5f5',
            textAlign: msg.role === 'user' ? 'right' : 'left'
          }}>
            <strong>{msg.role === 'user' ? 'You' : 'AI'}:</strong>
            <div style={{ marginTop: '4px', whiteSpace: 'pre-wrap' }}>
              {msg.content}
            </div>
          </div>
        ))}
        <div ref={messagesEndRef} />
      </div>

      <div style={{ display: 'flex', gap: '8px' }}>
        <input
          type="text"
          value={input}
          onChange={(e) => setInput(e.target.value)}
          onKeyPress={(e) => e.key === 'Enter' && sendMessage()}
          placeholder={connected ? "Ask about this stock..." : "Connecting..."}
          disabled={!connected}
          style={{
            flex: 1,
            padding: '12px',
            borderRadius: '4px',
            border: '1px solid #ccc'
          }}
        />
        <button
          onClick={sendMessage}
          disabled={!connected || !input.trim()}
          style={{
            padding: '12px 24px',
            borderRadius: '4px',
            border: 'none',
            backgroundColor: connected ? '#1976d2' : '#ccc',
            color: 'white',
            cursor: connected ? 'pointer' : 'not-allowed'
          }}
        >
          Send
        </button>
      </div>

      <div style={{ marginTop: '8px', fontSize: '12px', color: '#666' }}>
        Status: {connected ? '🟢 Connected' : '🔴 Disconnected'}
      </div>
    </div>
  )
}
```

```typescript
// src/components/TickerInput.tsx
import { useState } from 'react'

export function TickerInput({ onTickerSubmit }: { onTickerSubmit: (ticker: string) => void }) {
  const [ticker, setTicker] = useState('')

  const handleSubmit = (e: React.FormEvent) => {
    e.preventDefault()
    if (ticker.trim()) {
      onTickerSubmit(ticker.toUpperCase())
    }
  }

  return (
    <form onSubmit={handleSubmit} style={{
      display: 'flex',
      gap: '12px',
      marginBottom: '24px',
      padding: '24px',
      backgroundColor: '#f5f5f5',
      borderRadius: '8px'
    }}>
      <input
        type="text"
        value={ticker}
        onChange={(e) => setTicker(e.target.value.toUpperCase())}
        placeholder="Enter ticker (e.g., AAPL)"
        style={{
          flex: 1,
          padding: '16px',
          fontSize: '18px',
          borderRadius: '4px',
          border: '2px solid #1976d2'
        }}
      />
      <button
        type="submit"
        style={{
          padding: '16px 32px',
          fontSize: '18px',
          borderRadius: '4px',
          border: 'none',
          backgroundColor: '#1976d2',
          color: 'white',
          cursor: 'pointer',
          fontWeight: 'bold'
        }}
      >
        Analyze
      </button>
    </form>
  )
}
```

### Step 3: Update Main App (30 min)

```typescript
// src/App.tsx
import { useState } from 'react'
import { ChatInterface } from './components/ChatInterface'
import { TickerInput } from './components/TickerInput'
import './App.css'

function App() {
  const [ticker, setTicker] = useState<string | null>(null)

  return (
    <div style={{
      maxWidth: '1200px',
      margin: '0 auto',
      padding: '24px'
    }}>
      <header style={{
        textAlign: 'center',
        marginBottom: '32px'
      }}>
        <h1 style={{ fontSize: '36px', marginBottom: '8px' }}>
          📊 AI Stock Analyst
        </h1>
        <p style={{ color: '#666' }}>
          Get instant AI-powered analysis of any stock
        </p>
      </header>

      <TickerInput onTickerSubmit={setTicker} />

      {ticker && <ChatInterface ticker={ticker} />}

      {!ticker && (
        <div style={{
          textAlign: 'center',
          padding: '48px',
          color: '#666'
        }}>
          <p>Enter a stock ticker to start analyzing</p>
        </div>
      )}

      <footer style={{
        marginTop: '48px',
        textAlign: 'center',
        color: '#999',
        fontSize: '12px'
      }}>
        <p>⚠️ For educational purposes only. Not financial advice.</p>
      </footer>
    </div>
  )
}

export default App
```

### Step 4: Test Frontend (15 min)

```bash
# Make sure backend is running on :8000
# Start frontend
cd frontend
npm run dev

# Open http://localhost:5173
# Try ticker: AAPL
# Send chat: "What are the main risks?"
```

**✅ Saturday Afternoon Checkpoint:** Chat UI working with backend

---

## 🔗 Saturday Evening: Integration (3 hours)

### Fix CORS Issues (if any)

```bash
# If you get CORS errors, update backend:
# backend/app/main.py - already has CORS configured
```

### Add Loading States (1 hour)

```typescript
// Update ChatInterface.tsx to show loading
const [loading, setLoading] = useState(false)

const sendMessage = () => {
  // ... existing code
  setLoading(true)
  socket.send(input)
}

// In onmessage handler:
ws.onmessage = (event) => {
  setLoading(false)
  // ... rest of code
}

// Add to UI:
{loading && <div>AI is thinking...</div>}
```

### Test End-to-End (1 hour)

Test these scenarios:
- [ ] Enter ticker AAPL
- [ ] Ask "What are the risks?"
- [ ] Ask "How is the stock performing?"
- [ ] Try another ticker: TSLA
- [ ] Refresh page (should work)

### Quick Polish (1 hour)

```css
/* src/App.css - Make it look better */
* {
  box-sizing: border-box;
  margin: 0;
  padding: 0;
}

body {
  font-family: system-ui, -apple-system, sans-serif;
  background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
  min-height: 100vh;
}

#root {
  min-height: 100vh;
  padding: 20px;
}
```

**✅ Saturday Evening Checkpoint:** Fully working app locally

---

## 🚀 Sunday Morning: Deploy to Railway (3 hours)

### Step 1: Prepare for Deployment (30 min)

```bash
# backend/Dockerfile
FROM python:3.12-slim

WORKDIR /app

# Copy requirements
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Copy code
COPY . .

# Copy parent directory code
COPY ../agents ./agents
COPY ../database ./database

EXPOSE 8000

CMD ["uvicorn", "app.main:app", "--host", "0.0.0.0", "--port", "8000"]
```

```bash
# frontend: Update vite.config.ts for production
import { defineConfig } from 'vite'
import react from '@vitejs/plugin-react'

export default defineConfig({
  plugins: [react()],
  server: {
    port: 3000
  }
})
```

### Step 2: Deploy Backend to Railway (1 hour)

```bash
# Install Railway CLI
npm install -g @railway/cli

# Login
railway login

# Initialize project
cd backend
railway init

# Add PostgreSQL (for later)
railway add postgresql

# Set environment variables
railway variables set GOOGLE_API_KEY="your-key"
railway variables set SEC_HEADER="your-email@example.com MyApp"

# Deploy!
railway up

# Get URL
railway domain
# Example: https://analyst-agent-production.up.railway.app
```

### Step 3: Deploy Frontend to Vercel (30 min)

```bash
# Install Vercel CLI
npm install -g vercel

# Update API URL for production
cd frontend

# Create .env.production
echo "VITE_API_URL=https://your-railway-app.up.railway.app" > .env.production

# Deploy
vercel

# Production deploy
vercel --prod
```

### Step 4: Update WebSocket URL (30 min)

```typescript
// frontend/src/components/ChatInterface.tsx
// Update WebSocket connection
const API_URL = import.meta.env.VITE_API_URL || 'http://localhost:8000'
const WS_URL = API_URL.replace('https://', 'wss://').replace('http://', 'ws://')

const ws = new WebSocket(`${WS_URL}/ws/chat/${ticker}`)
```

**✅ Sunday Morning Checkpoint:** App deployed and live!

---

## 🐛 Sunday Afternoon: Test & Fix (5 hours)

### Testing Checklist

- [ ] Open live URL on desktop
- [ ] Try 5 different tickers
- [ ] Ask 3 questions per ticker
- [ ] Test on mobile (responsive?)
- [ ] Test WebSocket reconnection
- [ ] Check error handling
- [ ] Monitor Railway logs

### Common Issues & Fixes

**Issue: WebSocket not connecting**
```python
# backend/app/main.py - Add WebSocket CORS
from fastapi.middleware.cors import CORSMiddleware

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # For MVP - restrict in production
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)
```

**Issue: Slow responses**
```python
# Add timeout to LLM calls
llm = ChatGoogleGenerativeAI(
    model="gemini-2.0-flash-exp",
    timeout=30,  # 30 second timeout
)
```

**Issue: Memory leak (agents piling up)**
```python
# Limit cache size
MAX_AGENTS = 10
if len(agent_cache) > MAX_AGENTS:
    # Remove oldest
    oldest_ticker = list(agent_cache.keys())[0]
    del agent_cache[oldest_ticker]
```

### Quick Wins

1. **Add Error Messages**
```typescript
{error && (
  <div style={{ color: 'red', padding: '12px' }}>
    Error: {error}
  </div>
)}
```

2. **Add Examples**
```typescript
<div>
  <p>Try asking:</p>
  <ul>
    <li>"What are the main risks?"</li>
    <li>"How is revenue growing?"</li>
    <li>"Analyze the balance sheet"</li>
  </ul>
</div>
```

3. **Add Disclaimer**
```typescript
<div style={{ backgroundColor: '#fff3cd', padding: '12px', borderRadius: '4px' }}>
  ⚠️ This is for educational purposes only. Not financial advice.
  Data may be delayed or inaccurate.
</div>
```

**✅ Sunday Afternoon Checkpoint:** App stable and usable

---

## 🎉 Sunday Evening: Launch (1 hour)

### Final Checklist

- [ ] Test on 3 different browsers
- [ ] Test on mobile
- [ ] Verify all links work
- [ ] Check logs for errors
- [ ] Set up basic monitoring (Railway dashboard)

### Share Your MVP

Post your live URL:
- Twitter/X
- LinkedIn
- Reddit (r/SideProject)
- HackerNews Show HN
- Friends & family

**Template post:**
```
Just shipped AI Stock Analyst in 48 hours! 🚀

Built with:
- FastAPI backend
- React frontend
- Google Gemini for AI
- Deployed on Railway

Try it: [your-url]

Feedback welcome! This is an MVP - built in a weekend.

#buildinpublic #AI #startup
```

---

## 📊 What You'll Have By Sunday 8pm

✅ **Live web app** at your-app.vercel.app
✅ **AI chat** for stock analysis
✅ **Working demo** to show investors/users
✅ **Shareable URL** for feedback
✅ **Foundation** to build on next week

## 🚧 What to Add Monday-Friday

**Monday:** Authentication (Clerk/Auth0 - 2 hours)
**Tuesday:** PostgreSQL + save chats (4 hours)
**Wednesday:** Better UI (Tailwind components - 4 hours)
**Thursday:** Charts (TradingView - 4 hours)
**Friday:** Export PDF (2 hours) + Testing (2 hours)

---

## 💡 Time-Saving Tips

1. **Copy-paste liberally** - Don't write from scratch
2. **Use Claude/Cursor** - Generate boilerplate
3. **Skip optimization** - Make it work first
4. **Use console.log** - Skip proper debugging
5. **Ignore warnings** - Ship first, fix later
6. **Mobile-first** - Test on phone only
7. **One browser** - Chrome only for MVP

---

## 🆘 Emergency Shortcuts

**If running behind schedule:**

**Saturday:** Skip WebSocket, use polling instead
```typescript
// Simple polling every 5 seconds
useEffect(() => {
  const interval = setInterval(async () => {
    const res = await fetch(`/api/v1/status/${ticker}`)
    // Update UI
  }, 5000)
}, [])
```

**Sunday:** Deploy backend only, skip frontend
```bash
# Just share the Railway API URL
# Users can use curl or Postman
curl -X POST https://your-app.up.railway.app/api/v1/analyze \
  -d '{"ticker": "AAPL"}'
```

---

## 📈 Success Metrics

By Sunday 8pm:
- [ ] App loads in < 3 seconds
- [ ] Can analyze 3 tickers successfully
- [ ] Chat works 80% of the time
- [ ] Deployed to public URL
- [ ] Shared with 5 people

**Good enough to ship! 🚀**

---

## 🎯 Remember

"Done is better than perfect"
"Ship it, then improve it"
"Weekend is for shipping, not perfecting"

You've got this! Start Saturday 9am sharp.
See you on the other side with a live product! 💪
