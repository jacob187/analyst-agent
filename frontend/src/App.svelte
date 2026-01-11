<script lang="ts">
  import ApiKeyInput from './lib/components/ApiKeyInput.svelte';
  import TickerInput from './lib/components/TickerInput.svelte';
  import ChatWindow from './lib/components/ChatWindow.svelte';

  let googleApiKey: string | null = null;
  let secHeader: string | null = null;
  let currentTicker: string | null = null;

  function handleApiKeySubmit(event: CustomEvent<{ googleApiKey: string; secHeader: string }>) {
    googleApiKey = event.detail.googleApiKey;
    secHeader = event.detail.secHeader;
  }

  function handleTickerSubmit(event: CustomEvent<string>) {
    currentTicker = event.detail;
  }

  function resetSession() {
    currentTicker = null;
    googleApiKey = null;
    secHeader = null;
  }
</script>

<div class="app-container">
  <header>
    <div class="logo">
      <span class="bracket">[</span>
      <span class="title">ANALYST</span>
      <span class="bracket">]</span>
    </div>
    <div class="subtitle">AI-Powered Financial Terminal</div>
  </header>

  <main>
    {#if !googleApiKey || !secHeader}
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

  @media (max-width: 768px) {
    .app-container {
      padding: 1rem;
    }

    .logo {
      font-size: 1.5rem;
    }

    .welcome {
      padding: 2rem 1rem;
    }

    .examples {
      flex-direction: column;
      width: 100%;
      max-width: 300px;
    }
  }
</style>
