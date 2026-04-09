<script lang="ts">
  import { onMount } from 'svelte';
  import ApiKeyInput from './lib/components/ApiKeyInput.svelte';
  import TickerInput from './lib/components/TickerInput.svelte';
  import ChatWindow from './lib/components/ChatWindow.svelte';
  import ChatHistory from './lib/components/ChatHistory.svelte';
  import ChatViewer from './lib/components/ChatViewer.svelte';
  import AboutPage from './lib/components/about/AboutPage.svelte';
  import StockChart from './lib/components/StockChart.svelte';
  import Watchlist from './lib/components/Watchlist.svelte';
  import CompanyDashboard from './lib/components/CompanyDashboard.svelte';

  const API_BASE = import.meta.env.VITE_API_URL || 'http://localhost:8000';

  type Page = 'main' | 'about' | 'history' | 'view-session' | 'continue-session' | 'settings' | 'watchlist' | 'company-profile';
  let currentPage: Page = 'main';
  let settingsLoaded = false;
  let menuOpen = false;

  let googleApiKey: string | null = null;
  let openaiApiKey: string | null = null;
  let anthropicApiKey: string | null = null;
  let secHeader: string | null = null;
  let tavilyApiKey: string = '';
  let selectedModelId: string = '';
  let models: any[] = [];
  let currentTicker: string | null = null;

  // For viewing/continuing past sessions
  let viewingSessionId: string | null = null;
  let viewingSessionTicker: string | null = null;
  let continuingSessionId: string | null = null;

  // Which keys are available in the server's environment (.env / local dev)
  let envKeys = { google: false, openai: false, anthropic: false, sec_header: false };

  // True if any LLM key is set — either in localStorage or in server env vars
  $: hasLlmKey = !!(googleApiKey || openaiApiKey || anthropicApiKey || envKeys.google || envKeys.openai || envKeys.anthropic);
  $: hasSecHeader = !!(secHeader || envKeys.sec_header);

  onMount(async () => {
    googleApiKey = localStorage.getItem('google_api_key');
    openaiApiKey = localStorage.getItem('openai_api_key');
    anthropicApiKey = localStorage.getItem('anthropic_api_key');
    secHeader = localStorage.getItem('sec_header');
    tavilyApiKey = localStorage.getItem('tavily_api_key') || '';
    selectedModelId = localStorage.getItem('model_id') || '';
    settingsLoaded = true;

    // Fetch available models and server-side env key availability in parallel
    const [modelsRes, envKeysRes] = await Promise.allSettled([
      fetch(`${API_BASE}/models`),
      fetch(`${API_BASE}/env-keys`),
    ]);

    if (modelsRes.status === 'fulfilled' && modelsRes.value.ok) {
      const data = await modelsRes.value.json();
      models = data.models || [];
      if (!selectedModelId && models.length > 0) {
        const defaultModel = models.find((m: any) => m.default) || models[0];
        selectedModelId = defaultModel.id;
      }
    } else {
      console.error('Failed to fetch models');
    }

    if (envKeysRes.status === 'fulfilled' && envKeysRes.value.ok) {
      envKeys = await envKeysRes.value.json();
    }
  });

  function closeMenu() { menuOpen = false; }
  function toggleMenu() { menuOpen = !menuOpen; }

  function navigateToAbout() { currentPage = 'about'; closeMenu(); }
  function navigateToMain() {
    currentPage = 'main';
    viewingSessionId = null;
    viewingSessionTicker = null;
    closeMenu();
  }
  function navigateToHistory() { currentPage = 'history'; closeMenu(); }
  function navigateToSettings() { currentPage = 'settings'; closeMenu(); }
  function navigateToWatchlist() { currentPage = 'watchlist'; closeMenu(); }

  function handleWatchlistSelect(event: CustomEvent<string>) {
    currentTicker = event.detail.toUpperCase();
    currentPage = 'company-profile';
  }

  function handleSessionSelect(event: CustomEvent<{ sessionId: string; ticker: string }>) {
    viewingSessionId = event.detail.sessionId;
    viewingSessionTicker = event.detail.ticker;
    currentPage = 'view-session';
  }

  function handleSessionContinue(event: CustomEvent<{ sessionId: string; ticker: string }>) {
    continuingSessionId = event.detail.sessionId;
    currentTicker = event.detail.ticker;
    currentPage = 'continue-session';
  }

  function handleApiKeySubmit(event: CustomEvent<{
    googleApiKey: string;
    openaiApiKey: string;
    anthropicApiKey: string;
    secHeader: string;
    tavilyApiKey: string;
    modelId: string;
  }>) {
    googleApiKey = event.detail.googleApiKey;
    openaiApiKey = event.detail.openaiApiKey;
    anthropicApiKey = event.detail.anthropicApiKey;
    secHeader = event.detail.secHeader;
    tavilyApiKey = event.detail.tavilyApiKey || '';
    selectedModelId = event.detail.modelId;

    const keyMap: Record<string, string> = {
      google_api_key: googleApiKey,
      openai_api_key: openaiApiKey,
      anthropic_api_key: anthropicApiKey,
      sec_header: secHeader,
      model_id: selectedModelId,
    };
    for (const [key, value] of Object.entries(keyMap)) {
      if (value) {
        localStorage.setItem(key, value);
      } else {
        localStorage.removeItem(key);
      }
    }
    if (tavilyApiKey) {
      localStorage.setItem('tavily_api_key', tavilyApiKey);
    } else {
      localStorage.removeItem('tavily_api_key');
    }

    if (currentPage === 'settings') {
      currentPage = 'main';
    }
  }

  async function handleTickerSubmit(event: CustomEvent<string>) {
    const ticker = event.detail.toUpperCase();
    currentTicker = ticker;

    try {
      const res = await fetch(`${API_BASE}/sessions/by-ticker/${ticker}`);
      if (res.ok) {
        const data = await res.json();
        if (data.session) {
          continuingSessionId = data.session.id;
          currentPage = 'continue-session';
          return;
        }
      }
    } catch (e) {
      console.error('Failed to check for existing session:', e);
    }
  }

  function resetSession() {
    currentTicker = null;
  }
</script>

<!-- Mobile overlay backdrop -->
{#if menuOpen}
  <!-- svelte-ignore a11y-click-events-have-key-events -->
  <!-- svelte-ignore a11y-no-static-element-interactions -->
  <div class="overlay" on:click={closeMenu}></div>
{/if}

<!-- Mobile slide-in drawer -->
<nav class="mobile-drawer" class:open={menuOpen} aria-hidden={!menuOpen}>
  <div class="drawer-header">
    <span class="drawer-logo"><span class="bracket">[</span>ANALYST<span class="bracket">]</span></span>
    <button class="drawer-close" on:click={closeMenu} aria-label="Close menu">✕</button>
  </div>
  <div class="drawer-nav">
    <button
      class="drawer-link"
      class:active={currentPage === 'main' || currentPage === 'continue-session' || currentPage === 'company-profile'}
      on:click={navigateToMain}
    >
      <span class="drawer-icon">⌨</span> Terminal
    </button>
    <button
      class="drawer-link"
      class:active={currentPage === 'watchlist'}
      on:click={navigateToWatchlist}
    >
      <span class="drawer-icon">◈</span> Watchlist
    </button>
    <button
      class="drawer-link"
      class:active={currentPage === 'history' || currentPage === 'view-session'}
      on:click={navigateToHistory}
    >
      <span class="drawer-icon">◷</span> History
    </button>
    <button
      class="drawer-link"
      class:active={currentPage === 'settings'}
      on:click={navigateToSettings}
    >
      <span class="drawer-icon">⚙</span> Settings
    </button>
    <button
      class="drawer-link"
      class:active={currentPage === 'about'}
      on:click={navigateToAbout}
    >
      <span class="drawer-icon">◎</span> About
    </button>
  </div>
  <div class="drawer-footer">
    <span class="drawer-disclaimer">Educational use only</span>
  </div>
</nav>

<div class="app-container">
  <header>
    <div class="header-content">
      <div class="logo">
        <span class="bracket">[</span>
        <span class="title">ANALYST</span>
        <span class="bracket">]</span>
      </div>

      <!-- Desktop navigation -->
      <nav class="nav-links">
        <button
          class="nav-link"
          class:active={currentPage === 'main' || currentPage === 'continue-session' || currentPage === 'company-profile'}
          on:click={navigateToMain}
        >
          Terminal
        </button>
        <button
          class="nav-link"
          class:active={currentPage === 'watchlist'}
          on:click={navigateToWatchlist}
        >
          Watchlist
        </button>
        <button
          class="nav-link"
          class:active={currentPage === 'history' || currentPage === 'view-session'}
          on:click={navigateToHistory}
        >
          History
        </button>
        <button
          class="nav-link"
          class:active={currentPage === 'settings'}
          on:click={navigateToSettings}
        >
          Settings
        </button>
        <button
          class="nav-link"
          class:active={currentPage === 'about'}
          on:click={navigateToAbout}
        >
          About
        </button>
      </nav>

      <!-- Mobile hamburger -->
      <button class="hamburger" on:click={toggleMenu} aria-label="Open menu" aria-expanded={menuOpen}>
        <span class="bar" class:open={menuOpen}></span>
        <span class="bar" class:open={menuOpen}></span>
        <span class="bar" class:open={menuOpen}></span>
      </button>
    </div>
    <div class="subtitle">AI-Powered Financial Terminal</div>
  </header>

  <main>
    {#if !settingsLoaded}
      <div class="loading">Loading...</div>
    {:else if currentPage === 'about'}
      <div class="scrollable-page">
        <AboutPage on:back={navigateToMain} />
      </div>
    {:else if currentPage === 'settings'}
      <div class="settings-page">
        <div class="settings-inner">
          <h2 class="page-title">Settings</h2>
          <ApiKeyInput
            googleApiKey={googleApiKey || ''}
            openaiApiKey={openaiApiKey || ''}
            anthropicApiKey={anthropicApiKey || ''}
            secHeader={secHeader || ''}
            tavilyApiKey={tavilyApiKey}
            selectedModelId={selectedModelId}
            {models}
            on:submit={handleApiKeySubmit}
          />
          <button class="btn" style="margin-top: 1rem;" on:click={navigateToMain}>
            ← back
          </button>
        </div>
      </div>
    {:else if currentPage === 'watchlist'}
      <Watchlist apiBase={API_BASE} on:select={handleWatchlistSelect} />
    {:else if currentPage === 'company-profile' && currentTicker}
      <CompanyDashboard
        ticker={currentTicker}
        apiBase={API_BASE}
        {googleApiKey}
        {openaiApiKey}
        {anthropicApiKey}
        {secHeader}
        {tavilyApiKey}
        modelId={selectedModelId}
        sessionId={continuingSessionId}
        on:back={navigateToWatchlist}
      />
    {:else if currentPage === 'history'}
      <div class="history-section">
        <ChatHistory on:select={handleSessionSelect} on:continue={handleSessionContinue} on:close={navigateToMain} />
      </div>
    {:else if currentPage === 'view-session' && viewingSessionId && viewingSessionTicker}
      <div class="chat-section">
        <ChatViewer sessionId={viewingSessionId} ticker={viewingSessionTicker} />
      </div>
      <div class="actions-bar">
        <button class="btn" on:click={navigateToHistory}>← back to history</button>
        <button class="btn" on:click={navigateToMain}>← new chat</button>
      </div>
    {:else if currentPage === 'continue-session' && continuingSessionId && currentTicker}
      {#if !hasLlmKey || !hasSecHeader}
        <div class="setup-prompt">
          <div class="setup-icon">⚙️</div>
          <h2>API Keys Required</h2>
          <p>Configure your API keys in Settings to continue the chat with <strong>{currentTicker}</strong></p>
          <button class="btn primary" on:click={navigateToSettings}>Go to Settings →</button>
        </div>
      {:else}
        <div class="terminal-layout">
          <div class="chart-column">
            <StockChart ticker={currentTicker} />
          </div>
          <div class="chat-column">
            <ChatWindow
              ticker={currentTicker}
              googleApiKey={googleApiKey}
              openaiApiKey={openaiApiKey}
              anthropicApiKey={anthropicApiKey}
              secHeader={secHeader}
              tavilyApiKey={tavilyApiKey}
              modelId={selectedModelId}
              sessionId={continuingSessionId}
            />
          </div>
        </div>
        <div class="actions-bar">
          <button class="btn" on:click={navigateToHistory}>← back to history</button>
          <button class="btn" on:click={navigateToMain}>← new chat</button>
        </div>
      {/if}
    {:else if !hasLlmKey || !hasSecHeader}
      <div class="setup-prompt">
        <div class="setup-icon">⚙️</div>
        <h2>Configure API Keys</h2>
        <p>Set up your API keys in Settings to start analyzing stocks</p>
        <button class="btn primary" on:click={navigateToSettings}>Go to Settings →</button>
      </div>
    {:else if !currentTicker}
      <div class="ticker-page">
        <div class="ticker-page-inner">
          <div class="welcome-text">
            <h2>Enter a ticker symbol to begin</h2>
            <p>Real-time SEC filings, technical analysis, and AI-powered insights</p>
          </div>
          <div class="ticker-wrap">
            <TickerInput on:submit={handleTickerSubmit} disabled={false} />
          </div>
          <div class="examples">
            <span>Try: AAPL</span>
            <span>TSLA</span>
            <span>MSFT</span>
          </div>
          <button class="btn secondary" on:click={navigateToSettings}>
            ⚙ update settings
          </button>
        </div>
      </div>
    {:else}
      <div class="terminal-layout">
        <div class="chart-column">
          <StockChart ticker={currentTicker} />
        </div>
        <div class="chat-column">
          <ChatWindow
            ticker={currentTicker}
            googleApiKey={googleApiKey}
            openaiApiKey={openaiApiKey}
            anthropicApiKey={anthropicApiKey}
            secHeader={secHeader}
            tavilyApiKey={tavilyApiKey}
            modelId={selectedModelId}
          />
        </div>
      </div>
      <div class="actions-bar">
        <button class="btn" on:click={() => currentTicker = null}>← new ticker</button>
        <button class="btn" on:click={navigateToSettings}>⚙ settings</button>
      </div>
    {/if}
  </main>

  <footer>
    <div class="disclaimer">⚠ For educational purposes only. Not financial advice.</div>
  </footer>
</div>

<style>
  /* ── Overlay ─────────────────────────────────────────────────────────── */
  .overlay {
    position: fixed;
    inset: 0;
    background: rgba(0, 0, 0, 0.6);
    backdrop-filter: blur(2px);
    z-index: 99;
    animation: fadeIn 0.2s ease;
  }

  @keyframes fadeIn {
    from { opacity: 0; }
    to   { opacity: 1; }
  }

  /* ── Mobile drawer ───────────────────────────────────────────────────── */
  .mobile-drawer {
    position: fixed;
    top: 0;
    right: 0;
    width: 270px;
    height: 100dvh;
    background: var(--bg-card);
    border-left: 1px solid var(--border);
    z-index: 100;
    display: flex;
    flex-direction: column;
    transform: translateX(100%);
    transition: transform 0.3s cubic-bezier(0.4, 0, 0.2, 1);
    /* Always rendered but hidden via transform — screen readers use aria-hidden */
  }

  .mobile-drawer.open {
    transform: translateX(0);
    box-shadow: -4px 0 24px rgba(0, 0, 0, 0.5);
  }

  .drawer-header {
    display: flex;
    align-items: center;
    justify-content: space-between;
    padding: 1.2rem 1.5rem;
    border-bottom: 1px solid var(--border);
  }

  .drawer-logo {
    font-size: 1.1rem;
    font-weight: 700;
    letter-spacing: 0.1em;
    color: var(--text);
  }

  .drawer-close {
    background: none;
    border: none;
    color: var(--text-dim);
    font-size: 1.1rem;
    cursor: pointer;
    padding: 0.3rem 0.5rem;
    border-radius: 4px;
    transition: color 0.2s, background 0.2s;
    line-height: 1;
  }

  .drawer-close:hover {
    color: var(--text);
    background: rgba(255, 255, 255, 0.06);
  }

  .drawer-nav {
    flex: 1;
    display: flex;
    flex-direction: column;
    padding: 0.8rem 0;
    overflow-y: auto;
  }

  .drawer-link {
    display: flex;
    align-items: center;
    gap: 0.9rem;
    background: none;
    border: none;
    color: var(--text-dim);
    font-family: inherit;
    font-size: 0.95rem;
    letter-spacing: 0.04em;
    padding: 0.9rem 1.5rem;
    cursor: pointer;
    text-align: left;
    transition: color 0.2s, background 0.2s;
    border-left: 2px solid transparent;
  }

  .drawer-link:hover {
    color: var(--text);
    background: rgba(255, 255, 255, 0.04);
  }

  .drawer-link.active {
    color: var(--accent);
    border-left-color: var(--accent);
    background: rgba(0, 255, 157, 0.05);
  }

  .drawer-icon {
    font-size: 1rem;
    width: 1.2rem;
    text-align: center;
    opacity: 0.7;
  }

  .drawer-footer {
    padding: 1rem 1.5rem;
    border-top: 1px solid var(--border);
  }

  .drawer-disclaimer {
    font-size: 0.7rem;
    color: var(--text-muted);
    letter-spacing: 0.03em;
  }

  /* ── Hamburger button ────────────────────────────────────────────────── */
  .hamburger {
    display: none;
    flex-direction: column;
    justify-content: center;
    gap: 5px;
    background: none;
    border: none;
    cursor: pointer;
    padding: 0.4rem;
    border-radius: 4px;
    transition: background 0.2s;
  }

  .hamburger:hover {
    background: rgba(255, 255, 255, 0.06);
  }

  .bar {
    display: block;
    width: 22px;
    height: 2px;
    background: var(--text);
    border-radius: 2px;
    transition: transform 0.25s ease, opacity 0.25s ease;
    transform-origin: center;
  }

  /* Animate bars into an X when open */
  .bar:nth-child(1).open { transform: translateY(7px) rotate(45deg); }
  .bar:nth-child(2).open { opacity: 0; transform: scaleX(0); }
  .bar:nth-child(3).open { transform: translateY(-7px) rotate(-45deg); }

  /* ── App shell ───────────────────────────────────────────────────────── */
  .app-container {
    max-width: 1800px;
    margin: 0 auto;
    padding: 1.5rem 2rem;
    height: 100dvh;
    display: flex;
    flex-direction: column;
    overflow: hidden;
  }

  header {
    margin-bottom: 1.5rem;
    flex-shrink: 0;
  }

  .header-content {
    display: flex;
    align-items: center;
    justify-content: space-between;
    margin-bottom: 0.4rem;
  }

  .logo {
    font-size: 1.8rem;
    font-weight: 700;
    letter-spacing: 0.1em;
  }

  .bracket { color: var(--accent); }
  .title   { color: var(--text); }

  .subtitle {
    color: var(--text-dim);
    font-size: 0.85rem;
    letter-spacing: 0.05em;
  }

  /* ── Desktop nav ─────────────────────────────────────────────────────── */
  .nav-links {
    display: flex;
    gap: 1.5rem;
  }

  .nav-link {
    background: none;
    border: none;
    color: var(--text-dim);
    font-family: inherit;
    font-size: 0.9rem;
    cursor: pointer;
    padding: 0.3rem 0;
    transition: color 0.2s;
    letter-spacing: 0.05em;
  }

  .nav-link:hover  { color: var(--accent); }
  .nav-link.active {
    color: var(--accent);
    border-bottom: 1px solid var(--accent);
  }

  /* ── Main content area ───────────────────────────────────────────────── */
  main {
    flex: 1;
    display: flex;
    flex-direction: column;
    gap: 1rem;
    min-height: 0;
    overflow: hidden;
  }

  /* ── Scrollable page (About, etc.) ──────────────────────────────────── */
  .scrollable-page {
    flex: 1;
    overflow-y: auto;
    min-height: 0;
    padding: 1rem 0;
  }

  /* ── Settings page — centered ────────────────────────────────────────── */
  .settings-page {
    flex: 1;
    display: flex;
    align-items: center;
    justify-content: center;
    padding: 1rem;
    overflow-y: auto;
  }

  .settings-inner {
    width: 100%;
    max-width: 600px;
  }

  .page-title {
    color: var(--text);
    font-size: 1.3rem;
    margin-bottom: 1.5rem;
    font-weight: 600;
  }

  /* ── Ticker landing page ─────────────────────────────────────────────── */
  .ticker-page {
    flex: 1;
    display: flex;
    align-items: center;
    justify-content: center;
    padding: 2rem 1rem;
  }

  .ticker-page-inner {
    display: flex;
    flex-direction: column;
    align-items: center;
    gap: 1.5rem;
    width: 100%;
    max-width: 500px;
    text-align: center;
  }

  .welcome-text h2 {
    color: var(--text);
    font-size: 1.4rem;
    font-weight: 600;
    margin-bottom: 0.5rem;
  }

  .welcome-text p {
    color: var(--text-dim);
    font-size: 0.9rem;
  }

  .ticker-wrap {
    width: 100%;
  }

  /* ── Setup / empty states ────────────────────────────────────────────── */
  .setup-prompt {
    flex: 1;
    display: flex;
    flex-direction: column;
    align-items: center;
    justify-content: center;
    text-align: center;
    padding: 4rem 2rem;
    gap: 0.8rem;
  }

  .setup-icon {
    font-size: 3.5rem;
    margin-bottom: 0.5rem;
  }

  .setup-prompt h2 {
    color: var(--text);
    font-size: 1.4rem;
    font-weight: 600;
  }

  .setup-prompt p {
    color: var(--text-dim);
    font-size: 0.95rem;
    max-width: 380px;
    margin-bottom: 0.8rem;
  }

  .setup-prompt strong { color: var(--accent); }

  /* ── Terminal layout ─────────────────────────────────────────────────── */
  .terminal-layout {
    display: grid;
    grid-template-columns: 3fr 2fr;
    gap: 1rem;
    flex: 1;
    min-height: 0;
    overflow: hidden;
  }

  .chart-column {
    overflow-y: auto;
    min-height: 0;
  }

  .chat-column {
    overflow: hidden;
    min-height: 0;
    display: flex;
    flex-direction: column;
  }

  /* ── History / view-session ──────────────────────────────────────────── */
  .history-section {
    display: flex;
    justify-content: center;
    padding: 2rem 0;
    overflow-y: auto;
    flex: 1;
  }

  .chat-section {
    flex: 1;
    min-height: 500px;
    overflow: hidden;
  }

  /* ── Misc shared ─────────────────────────────────────────────────────── */
  .actions-bar {
    display: flex;
    gap: 0.8rem;
    flex-wrap: wrap;
    flex-shrink: 0;
  }

  .examples {
    display: flex;
    gap: 0.8rem;
    flex-wrap: wrap;
    justify-content: center;
  }

  .examples span {
    padding: 0.4rem 0.9rem;
    background: var(--bg-card);
    border: 1px solid var(--border);
    border-radius: 4px;
    color: var(--text-dim);
    font-size: 0.85rem;
  }

  .loading {
    color: var(--text-dim);
    text-align: center;
    padding: 4rem;
  }

  .btn {
    background: var(--bg-card);
    color: var(--text-dim);
    border: 1px solid var(--border);
    padding: 0.5rem 1rem;
    border-radius: 4px;
    font-family: inherit;
    font-size: 0.85rem;
    cursor: pointer;
    transition: all 0.2s;
    letter-spacing: 0.02em;
  }

  .btn:hover {
    color: var(--text);
    border-color: var(--accent);
  }

  .btn.primary {
    background: var(--accent);
    color: var(--bg-dark);
    border-color: var(--accent);
    font-weight: 600;
  }

  .btn.primary:hover {
    background: var(--accent-dim);
    border-color: var(--accent-dim);
  }

  .btn.secondary {
    color: var(--text-muted);
    font-size: 0.8rem;
  }

  footer {
    margin-top: 1rem;
    padding-top: 1rem;
    border-top: 1px solid var(--border);
    flex-shrink: 0;
  }

  .disclaimer {
    text-align: center;
    color: var(--text-muted);
    font-size: 0.75rem;
  }

  /* ── Tablet (≤ 768px) ────────────────────────────────────────────────── */
  @media (max-width: 768px) {
    .app-container {
      padding: 1rem;
      height: auto;
      min-height: 100dvh;
      overflow-y: auto;
    }

    header {
      margin-bottom: 1rem;
    }

    .logo { font-size: 1.4rem; }

    /* Hide desktop nav, show hamburger */
    .nav-links  { display: none; }
    .hamburger  { display: flex; }

    main {
      overflow: visible;
    }

    /* Stack chart above chat on tablet */
    .terminal-layout {
      grid-template-columns: 1fr;
      overflow: visible;
      height: auto;
      flex: none;
    }

    .chart-column {
      height: auto;
      max-height: 45vw;
      min-height: 220px;
      overflow-y: auto;
    }

    .chat-column {
      height: 55vh;
      min-height: 320px;
      overflow: hidden;
    }

    .chat-section {
      min-height: 400px;
    }

    .history-section {
      padding: 1rem 0;
    }

    .setup-prompt {
      padding: 3rem 1.5rem;
    }

    .settings-page {
      align-items: flex-start;
      padding: 1.5rem 0;
    }
  }

  /* ── Mobile (≤ 480px) ────────────────────────────────────────────────── */
  @media (max-width: 480px) {
    .app-container {
      padding: 0.75rem;
    }

    .logo { font-size: 1.2rem; }

    .subtitle { font-size: 0.75rem; }

    main { gap: 0.75rem; }

    .terminal-layout { gap: 0.75rem; }

    .chart-column {
      max-height: 40vw;
      min-height: 180px;
    }

    .chat-column {
      height: 50vh;
      min-height: 280px;
    }

    .chat-section { min-height: 350px; }

    .setup-prompt {
      padding: 2rem 1rem;
    }

    .setup-icon { font-size: 2.8rem; }

    .setup-prompt h2 { font-size: 1.2rem; }

    .ticker-page { padding: 1.5rem 0.5rem; }

    .welcome-text h2 { font-size: 1.15rem; }

    .examples { gap: 0.5rem; }

    .actions-bar { gap: 0.6rem; }

    footer {
      margin-top: 0.75rem;
      padding-top: 0.75rem;
    }

    .disclaimer { font-size: 0.7rem; }
  }
</style>
