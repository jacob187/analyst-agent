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

  const API_BASE = import.meta.env.VITE_API_URL || 'http://localhost:8000';

  type Page = 'main' | 'about' | 'history' | 'view-session' | 'continue-session' | 'settings' | 'watchlist';
  let currentPage: Page = 'main';
  let settingsLoaded = false;

  let googleApiKey: string | null = null;
  let secHeader: string | null = null;
  let tavilyApiKey: string = '';
  let currentTicker: string | null = null;

  // For viewing/continuing past sessions
  let viewingSessionId: string | null = null;
  let viewingSessionTicker: string | null = null;
  let continuingSessionId: string | null = null;

  onMount(() => {
    googleApiKey = localStorage.getItem('google_api_key');
    secHeader = localStorage.getItem('sec_header');
    tavilyApiKey = localStorage.getItem('tavily_api_key') || '';
    settingsLoaded = true;
  });

  function navigateToAbout() {
    currentPage = 'about';
  }

  function navigateToMain() {
    currentPage = 'main';
    viewingSessionId = null;
    viewingSessionTicker = null;
  }

  function navigateToHistory() {
    currentPage = 'history';
  }

  function navigateToSettings() {
    currentPage = 'settings';
  }

  function navigateToWatchlist() {
    currentPage = 'watchlist';
  }

  function handleWatchlistSelect(event: CustomEvent<string>) {
    currentTicker = event.detail;
    // Route through the same ticker-submission flow to reuse session logic
    handleTickerSubmit(new CustomEvent('submit', { detail: event.detail }));
  }

  function handleSessionSelect(event: CustomEvent<{ sessionId: string; ticker: string }>) {
    viewingSessionId = event.detail.sessionId;
    viewingSessionTicker = event.detail.ticker;
    currentPage = 'view-session';
  }

  function handleSessionContinue(event: CustomEvent<{ sessionId: string; ticker: string }>) {
    continuingSessionId = event.detail.sessionId;
    currentTicker = event.detail.ticker;
    // If no API keys yet, go to config first
    if (!googleApiKey || !secHeader) {
      currentPage = 'continue-session';
    } else {
      currentPage = 'continue-session';
    }
  }

  function handleApiKeySubmit(event: CustomEvent<{ googleApiKey: string; secHeader: string; tavilyApiKey: string }>) {
    googleApiKey = event.detail.googleApiKey;
    secHeader = event.detail.secHeader;
    tavilyApiKey = event.detail.tavilyApiKey || '';

    localStorage.setItem('google_api_key', googleApiKey);
    localStorage.setItem('sec_header', secHeader);
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

    // Check whether a session already exists for this ticker.
    // If so, route into the "continue" flow so past messages load in the UI.
    // To start completely fresh, the user must delete the session from History first.
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
    // No existing session — currentTicker is set, template renders new ChatWindow
  }

  function resetSession() {
    currentTicker = null;
  }
</script>

<div class="app-container">
  <header>
    <div class="header-content">
      <div class="logo">
        <span class="bracket">[</span>
        <span class="title">ANALYST</span>
        <span class="bracket">]</span>
      </div>
      <nav class="nav-links">
        <button
          class="nav-link"
          class:active={currentPage === 'main' || currentPage === 'continue-session'}
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
    </div>
    <div class="subtitle">AI-Powered Financial Terminal</div>
  </header>

  <main>
    {#if !settingsLoaded}
      <div class="loading">Loading...</div>
    {:else if currentPage === 'about'}
      <AboutPage on:back={navigateToMain} />
    {:else if currentPage === 'settings'}
      <div class="config-section">
        <h2 class="page-title">Settings</h2>
        <ApiKeyInput
          googleApiKey={googleApiKey || ''}
          secHeader={secHeader || ''}
          tavilyApiKey={tavilyApiKey}
          on:submit={handleApiKeySubmit}
        />
        <button class="btn" style="margin-top: 1rem;" on:click={navigateToMain}>
          ← back
        </button>
      </div>
    {:else if currentPage === 'watchlist'}
      <Watchlist apiBase={API_BASE} on:select={handleWatchlistSelect} />
    {:else if currentPage === 'history'}
      <div class="history-section">
        <ChatHistory on:select={handleSessionSelect} on:continue={handleSessionContinue} on:close={navigateToMain} />
      </div>
    {:else if currentPage === 'view-session' && viewingSessionId && viewingSessionTicker}
      <div class="chat-section">
        <ChatViewer sessionId={viewingSessionId} ticker={viewingSessionTicker} />
      </div>
      <div class="actions-bar">
        <button class="btn" on:click={navigateToHistory}>
          ← back to history
        </button>
        <button class="btn" on:click={navigateToMain}>
          ← new chat
        </button>
      </div>
    {:else if currentPage === 'continue-session' && continuingSessionId && currentTicker}
      {#if !googleApiKey || !secHeader}
        <!-- Need API keys to continue - point to Settings -->
        <div class="setup-prompt">
          <div class="setup-icon">⚙️</div>
          <h2>API Keys Required</h2>
          <p>Configure your API keys in Settings to continue the chat with <strong>{currentTicker}</strong></p>
          <button class="btn primary" on:click={navigateToSettings}>
            Go to Settings →
          </button>
        </div>
      {:else}
        <!-- Ready to continue — Chart + Chat Split Layout -->
        <div class="terminal-layout">
          <div class="chart-column">
            <StockChart ticker={currentTicker} />
          </div>
          <div class="chat-column">
            <ChatWindow
              ticker={currentTicker}
              googleApiKey={googleApiKey}
              secHeader={secHeader}
              tavilyApiKey={tavilyApiKey}
              sessionId={continuingSessionId}
            />
          </div>
        </div>
        <div class="actions-bar">
          <button class="btn" on:click={navigateToHistory}>
            ← back to history
          </button>
          <button class="btn" on:click={navigateToMain}>
            ← new chat
          </button>
        </div>
      {/if}
    {:else if !googleApiKey || !secHeader}
      <!-- No API keys configured - point to Settings -->
      <div class="setup-prompt">
        <div class="setup-icon">⚙️</div>
        <h2>Configure API Keys</h2>
        <p>Set up your API keys in Settings to start analyzing stocks</p>
        <button class="btn primary" on:click={navigateToSettings}>
          Go to Settings →
        </button>
      </div>
    {:else if !currentTicker}
      <!-- Step 2: Ticker Selection -->
      <div class="input-section">
        <TickerInput on:submit={handleTickerSubmit} disabled={false} />
      </div>
      <div class="welcome">
        <div class="icon">📊</div>
        <h2>Enter a ticker symbol to begin</h2>
        <p>Get real-time SEC filings, technical analysis, and AI-powered insights</p>
        <div class="examples">
          <span>Try: AAPL</span>
          <span>TSLA</span>
          <span>MSFT</span>
        </div>
      </div>
      <button class="btn" on:click={navigateToSettings}>
        ← update API keys
      </button>
    {:else}
      <!-- Step 3: Chart + Chat Split Layout -->
      <div class="terminal-layout">
        <div class="chart-column">
          <StockChart ticker={currentTicker} />
        </div>
        <div class="chat-column">
          <ChatWindow
            ticker={currentTicker}
            googleApiKey={googleApiKey}
            secHeader={secHeader}
            tavilyApiKey={tavilyApiKey}
          />
        </div>
      </div>
      <div class="actions-bar">
        <button class="btn" on:click={() => currentTicker = null}>
          ← new ticker
        </button>
        <button class="btn" on:click={navigateToSettings}>
          ← settings
        </button>
      </div>
    {/if}
  </main>

  <footer>
    <div class="disclaimer">
      ⚠ For educational purposes only. Not financial advice.
    </div>
  </footer>
</div>

<style>
  .app-container {
    max-width: 1800px;
    margin: 0 auto;
    padding: 2rem;
    height: 100vh;
    display: flex;
    flex-direction: column;
    overflow: hidden;
  }

  header {
    margin-bottom: 2rem;
  }

  .logo {
    font-size: 2rem;
    font-weight: 700;
    letter-spacing: 0.1em;
    margin-bottom: 0.5rem;
  }

  .bracket {
    color: var(--accent);
  }

  .title {
    color: var(--text);
  }

  .subtitle {
    color: var(--text-dim);
    font-size: 0.9rem;
    letter-spacing: 0.05em;
  }

  .header-content {
    display: flex;
    align-items: center;
    justify-content: space-between;
    margin-bottom: 0.5rem;
  }

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

  .nav-link:hover {
    color: var(--accent);
  }

  .nav-link.active {
    color: var(--accent);
    border-bottom: 1px solid var(--accent);
  }

  main {
    flex: 1;
    display: flex;
    flex-direction: column;
    gap: 1.5rem;
    min-height: 0; /* prevent flex child from expanding beyond parent */
    overflow-y: auto;
  }

  .config-section {
    width: 100%;
    max-width: 600px;
  }

  .loading {
    color: var(--text-dim);
    text-align: center;
    padding: 4rem;
  }

  .page-title {
    color: var(--text);
    font-size: 1.3rem;
    margin-bottom: 1.5rem;
    font-weight: 600;
  }

  .setup-prompt {
    flex: 1;
    display: flex;
    flex-direction: column;
    align-items: center;
    justify-content: center;
    text-align: center;
    padding: 4rem 2rem;
  }

  .setup-icon {
    font-size: 4rem;
    margin-bottom: 1.5rem;
  }

  .setup-prompt h2 {
    color: var(--text);
    font-size: 1.5rem;
    margin-bottom: 0.8rem;
    font-weight: 600;
  }

  .setup-prompt p {
    color: var(--text-dim);
    font-size: 1rem;
    max-width: 400px;
    margin-bottom: 2rem;
  }

  .setup-prompt strong {
    color: var(--accent);
  }

  .btn.primary {
    background: var(--accent);
    color: var(--bg-dark);
    font-weight: 600;
  }

  .btn.primary:hover {
    background: var(--accent-dim);
  }

  .history-section {
    display: flex;
    justify-content: center;
    padding: 2rem 0;
  }

  .input-section {
    width: 100%;
    max-width: 600px;
  }

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

  .chat-section {
    flex: 1;
    min-height: 500px;
  }

  .actions-bar {
    display: flex;
    gap: 1rem;
    flex-wrap: wrap;
  }


  .welcome {
    flex: 1;
    display: flex;
    flex-direction: column;
    align-items: center;
    justify-content: center;
    text-align: center;
    padding: 4rem 2rem;
  }

  .icon {
    font-size: 4rem;
    margin-bottom: 1.5rem;
  }

  .welcome h2 {
    color: var(--text);
    font-size: 1.5rem;
    margin-bottom: 0.8rem;
    font-weight: 600;
  }

  .welcome p {
    color: var(--text-dim);
    font-size: 1rem;
    max-width: 500px;
    margin-bottom: 2rem;
  }

  .examples {
    display: flex;
    gap: 1rem;
  }

  .examples span {
    padding: 0.5rem 1rem;
    background: var(--bg-card);
    border: 1px solid var(--border);
    border-radius: 4px;
    color: var(--text-dim);
    font-size: 0.9rem;
  }

  footer {
    margin-top: 2rem;
    padding-top: 1.5rem;
    border-top: 1px solid var(--border);
  }

  .disclaimer {
    text-align: center;
    color: var(--text-muted);
    font-size: 0.8rem;
  }

  /* Tablet breakpoint */
  @media (max-width: 768px) {
    .app-container {
      padding: 1rem;
      height: auto;
      min-height: 100vh;
      overflow-y: auto;
    }

    .logo {
      font-size: 1.5rem;
    }

    .header-content {
      flex-direction: column;
      align-items: flex-start;
      gap: 0.5rem;
    }

    .nav-links {
      gap: 1rem;
    }

    .config-section,
    .input-section {
      max-width: 100%;
    }

    .terminal-layout {
      grid-template-columns: 1fr;
      overflow: visible;
      height: auto;
      flex: none;
    }

    .chart-column {
      height: auto;
      max-height: 50vh;
      overflow-y: auto;
    }

    .chat-column {
      height: 60vh;
      overflow: hidden;
    }

    .chat-section {
      min-height: 400px;
    }

    .welcome {
      padding: 2rem 1rem;
    }

    .icon {
      font-size: 3rem;
    }

    .welcome h2 {
      font-size: 1.3rem;
    }

    .welcome p {
      font-size: 0.9rem;
    }

    .examples {
      flex-direction: column;
      width: 100%;
      max-width: 300px;
    }

    .actions-bar {
      flex-direction: column;
    }

  }

  /* Small mobile breakpoint */
  @media (max-width: 480px) {
    .app-container {
      padding: 0.8rem;
    }

    header {
      margin-bottom: 1.5rem;
    }

    .logo {
      font-size: 1.3rem;
    }

    .subtitle {
      font-size: 0.75rem;
    }

    main {
      gap: 1rem;
    }

    .chart-column {
      max-height: 45vh;
    }

    .chat-column {
      height: 55vh;
    }

    .chat-section {
      min-height: 350px;
    }

    .welcome {
      padding: 1.5rem 0.5rem;
    }

    .icon {
      font-size: 2.5rem;
      margin-bottom: 1rem;
    }

    .welcome h2 {
      font-size: 1.1rem;
      margin-bottom: 0.5rem;
    }

    .welcome p {
      font-size: 0.85rem;
      margin-bottom: 1.5rem;
    }

    .examples {
      gap: 0.6rem;
    }

    .examples span {
      padding: 0.4rem 0.8rem;
      font-size: 0.85rem;
    }


    footer {
      margin-top: 1.5rem;
      padding-top: 1rem;
    }

    .disclaimer {
      font-size: 0.7rem;
    }
  }
</style>
