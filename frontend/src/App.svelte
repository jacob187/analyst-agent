<script lang="ts">
  import ApiKeyInput from './lib/components/ApiKeyInput.svelte';
  import TickerInput from './lib/components/TickerInput.svelte';
  import ChatWindow from './lib/components/ChatWindow.svelte';
  import AboutPage from './lib/components/about/AboutPage.svelte';

  type Page = 'main' | 'about';
  let currentPage: Page = 'main';

  let googleApiKey: string | null = null;
  let secHeader: string | null = null;
  let tavilyApiKey: string = '';
  let currentTicker: string | null = null;

  function navigateToAbout() {
    currentPage = 'about';
  }

  function navigateToMain() {
    currentPage = 'main';
  }

  function handleApiKeySubmit(event: CustomEvent<{ googleApiKey: string; secHeader: string; tavilyApiKey: string }>) {
    googleApiKey = event.detail.googleApiKey;
    secHeader = event.detail.secHeader;
    tavilyApiKey = event.detail.tavilyApiKey || '';
  }

  function handleTickerSubmit(event: CustomEvent<string>) {
    currentTicker = event.detail;
  }

  function resetSession() {
    currentTicker = null;
    googleApiKey = null;
    secHeader = null;
    tavilyApiKey = '';
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
      </button>
      <nav class="nav-links">
        <button
          class="nav-link"
          class:active={currentPage === 'main'}
          on:click={navigateToMain}
        >
          Terminal
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
    {#if currentPage === 'about'}
      <AboutPage on:back={navigateToMain} />
    {:else if !googleApiKey || !secHeader}
      <!-- Step 1: API Key Configuration -->
      <div class="config-section">
        <ApiKeyInput on:submit={handleApiKeySubmit} />
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
      <button class="reset-btn" on:click={resetSession}>
        ← reconfigure API keys
      </button>
    {:else}
      <!-- Step 3: Chat Interface -->
      <div class="chat-section">
        <ChatWindow
          ticker={currentTicker}
          googleApiKey={googleApiKey}
          secHeader={secHeader}
          tavilyApiKey={tavilyApiKey}
        />
      </div>
      <div class="actions-bar">
        <button class="reset-btn" on:click={() => currentTicker = null}>
          ← new ticker
        </button>
        <button class="reset-btn" on:click={resetSession}>
          ← reconfigure
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
    max-width: 1400px;
    margin: 0 auto;
    padding: 2rem;
    min-height: 100vh;
    display: flex;
    flex-direction: column;
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

  .logo-link {
    font-size: 2rem;
    font-weight: 700;
    letter-spacing: 0.1em;
    display: flex;
    align-items: center;
  }

  .logo-link:hover .bracket,
  .logo-link:hover .title {
    color: var(--accent);
  }

  main {
    flex: 1;
    display: flex;
    flex-direction: column;
    gap: 1.5rem;
  }

  .config-section {
    width: 100%;
    max-width: 600px;
  }

  .input-section {
    width: 100%;
    max-width: 600px;
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

  .reset-btn {
    background: var(--bg-card);
    color: var(--text-dim);
    border: 1px solid var(--border);
    padding: 0.6rem 1.2rem;
    border-radius: 4px;
    font-size: 0.85rem;
    cursor: pointer;
    transition: all 0.2s;
  }

  .reset-btn:hover {
    border-color: var(--accent);
    color: var(--accent);
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

    .reset-btn {
      width: 100%;
      text-align: center;
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

    .reset-btn {
      padding: 0.5rem 1rem;
      font-size: 0.8rem;
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
