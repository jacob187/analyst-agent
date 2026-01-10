# Render Deployment Guide

Deploy the Analyst Agent app to Render.

## Prerequisites

- GitHub account with your repo pushed
- Google Gemini API key ([get one here](https://makersuite.google.com/app/apikey))

## Quick Start

### 1. Push to GitHub

```bash
git checkout deploy
git push origin deploy
```

### 2. Deploy on Render

1. Go to [render.com](https://render.com) and sign in
2. Click **New** → **Blueprint**
3. Connect your GitHub repository
4. Select the `deploy` branch
5. Render will detect `render.yaml` and create both services

### 3. Set Environment Variables

In the Render dashboard, go to **analyst-agent-api** → **Environment**:

| Variable | Value |
|----------|-------|
| `GOOGLE_API_KEY` | Your Gemini API key |
| `SEC_HEADER` | `your.email@example.com YourName` |

> **Note:** `SEC_HEADER` is required by the SEC EDGAR API. Use format: `email@example.com Your Name`

### 4. Deploy

Click **Manual Deploy** → **Deploy latest commit**

### 5. Access the App

After deployment completes:
- **Frontend:** `https://analyst-agent-frontend.onrender.com`
- **API:** `https://analyst-agent-api.onrender.com`
- **Health check:** `https://analyst-agent-api.onrender.com/health`

## Services Created

| Service | Type | Plan |
|---------|------|------|
| `analyst-agent-api` | Web Service | Free (sleeps after 15min) |
| `analyst-agent-frontend` | Static Site | Free |

## Commands Reference

Using [Render CLI](https://render.com/docs/cli):

```bash
# Install CLI
brew install render

# Login
render login

# View services
render services list

# View logs
render logs analyst-agent-api
```

## Upgrading to Paid

To prevent cold starts (free tier sleeps after 15min idle):

1. Go to **analyst-agent-api** → **Settings**
2. Change plan from **Free** to **Starter** ($7/mo)

## Troubleshooting

### Build failed
- Check build logs in Render dashboard
- Ensure `uv.lock` is committed: `uv lock && git add uv.lock`

### WebSocket connection failed
- Verify frontend `VITE_API_URL` points to API service
- Check API logs for errors

### SEC API errors
- Verify `SEC_HEADER` format: `email@example.com YourName`

### Redeploy after code changes
```bash
git push origin deploy
```
Render auto-deploys on push, or use **Manual Deploy** in dashboard.
