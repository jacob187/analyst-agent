<script lang="ts">
  /**
   * Watchlist — manage tracked tickers and generate AI morning briefings.
   *
   * Uses Svelte 4 syntax (export let, on:click) to match the majority
   * of existing components in this project.
   */
  import { onMount, createEventDispatcher } from 'svelte';
  import { marked } from 'marked';
  import DOMPurify from 'dompurify';

  export let apiBase: string;

  const dispatch = createEventDispatcher<{ select: string }>();

  let tickers: Array<{ ticker: string; added_at: string; name?: string; sector?: string }> = [];
  let prices: Record<string, { price: number; change: number; changePercent: number }> = {};
  let newTicker = '';
  let briefing = '';
  let thinking = '';
  let briefingHistory: Array<{ id: string; market_regime: string; created_at: string; tickers: any[] }> = [];
  let showHistory = false;
  let loadingBriefing = false;
  let loadingHistory = false;
  let loadingList = true;
  let error = '';
  let addError = '';

  onMount(() => {
    fetchWatchlist();
  });

  async function fetchWatchlist() {
    loadingList = true;
    try {
      const res = await fetch(`${apiBase}/watchlist`);
      const data = await res.json();
      tickers = data.tickers || [];
      // Fetch prices for each ticker in parallel
      await Promise.all(tickers.map(t => fetchPrice(t.ticker)));
    } catch (e) {
      error = 'Failed to load watchlist';
    } finally {
      loadingList = false;
    }
  }

  async function fetchPrice(ticker: string) {
    try {
      const res = await fetch(`${apiBase}/stock/${ticker}/chart?period=1y&indicators=`);
      if (res.ok) {
        const data = await res.json();
        if (data.quote?.price) {
          prices[ticker] = {
            price: data.quote.price,
            change: data.quote.change,
            changePercent: data.quote.changePercent,
          };
          prices = prices; // trigger reactivity
        }
      }
    } catch {
      // Price fetch is best-effort
    }
  }

  async function addTicker() {
    const ticker = newTicker.trim().toUpperCase();
    if (!ticker) return;
    addError = '';

    try {
      const res = await fetch(`${apiBase}/watchlist`, {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify({ ticker }),
      });

      if (!res.ok) {
        const data = await res.json().catch(() => ({}));
        addError = data.detail || 'Failed to add ticker';
        return;
      }

      newTicker = '';
      await fetchWatchlist();
    } catch {
      addError = 'Failed to add ticker';
    }
  }

  async function removeTicker(ticker: string) {
    try {
      await fetch(`${apiBase}/watchlist/${ticker}`, { method: 'DELETE' });
      tickers = tickers.filter(t => t.ticker !== ticker);
      delete prices[ticker];
      prices = prices;
    } catch {
      // Silently fail — will resync on next load
    }
  }

  async function getBriefing() {
    loadingBriefing = true;
    briefing = '';
    thinking = '';
    error = '';

    try {
      const headers: Record<string, string> = {};
      const gKey = localStorage.getItem('google_api_key');
      const tKey = localStorage.getItem('tavily_api_key');
      if (gKey) headers['X-Google-Api-Key'] = gKey;
      if (tKey) headers['X-Tavily-Api-Key'] = tKey;

      const res = await fetch(`${apiBase}/watchlist/briefing`, { headers });
      if (!res.ok) {
        const data = await res.json().catch(() => ({}));
        error = data.detail || 'Failed to generate briefing';
        return;
      }
      const data = await res.json();
      briefing = data.briefing || '';
      thinking = data.thinking || '';
    } catch {
      error = 'Failed to generate briefing';
    } finally {
      loadingBriefing = false;
    }
  }

  async function fetchBriefingHistory() {
    loadingHistory = true;
    try {
      const res = await fetch(`${apiBase}/watchlist/briefing/history`);
      if (res.ok) {
        const data = await res.json();
        briefingHistory = data.briefings || [];
      }
    } catch {
      // History fetch is best-effort
    } finally {
      loadingHistory = false;
    }
  }

  function toggleHistory() {
    showHistory = !showHistory;
    if (showHistory && briefingHistory.length === 0) {
      fetchBriefingHistory();
    }
  }

  function formatDate(dateStr: string): string {
    try {
      const d = new Date(dateStr);
      return d.toLocaleDateString('en-US', { month: 'short', day: 'numeric', hour: '2-digit', minute: '2-digit' });
    } catch {
      return dateStr;
    }
  }

  function selectTicker(ticker: string) {
    dispatch('select', ticker);
  }

  function handleKeydown(e: KeyboardEvent) {
    if (e.key === 'Enter') addTicker();
  }

  function renderMarkdown(text: string): string {
    return DOMPurify.sanitize(marked.parse(text) as string);
  }
</script>

<div class="watchlist">
  <div class="watchlist-header">
    <h2>Watchlist</h2>
    <span class="count">{tickers.length} / 10</span>
  </div>

  <!-- Add ticker input -->
  <div class="add-row">
    <input
      type="text"
      bind:value={newTicker}
      on:keydown={handleKeydown}
      placeholder="Add ticker..."
      maxlength="10"
      class="add-input"
    />
    <button class="add-btn" on:click={addTicker} disabled={!newTicker.trim()}>
      Add
    </button>
  </div>
  {#if addError}
    <div class="add-error">{addError}</div>
  {/if}

  <!-- Ticker list -->
  {#if loadingList}
    <div class="loading-msg">Loading watchlist...</div>
  {:else if tickers.length === 0}
    <div class="empty-msg">No tickers in watchlist. Add some above to get started.</div>
  {:else}
    <div class="ticker-list">
      {#each tickers as t}
        {@const p = prices[t.ticker]}
        <div class="ticker-row">
          <div class="ticker-info">
            <button class="ticker-name" on:click={() => selectTicker(t.ticker)}>
              {t.ticker}
            </button>
            {#if t.name}
              <span class="company-name">{t.name}</span>
            {/if}
          </div>
          <div class="ticker-price">
            {#if p}
              <span class="price">${p.price.toFixed(2)}</span>
              <span class="change" class:up={p.change >= 0} class:down={p.change < 0}>
                {p.change >= 0 ? '+' : ''}{p.changePercent.toFixed(2)}%
              </span>
            {:else}
              <span class="price dim">--</span>
            {/if}
          </div>
          <button class="remove-btn" on:click={() => removeTicker(t.ticker)}>
            x
          </button>
        </div>
      {/each}
    </div>
  {/if}

  <!-- Briefing section -->
  {#if tickers.length > 0}
    <div class="briefing-section">
      <button
        class="briefing-btn"
        on:click={getBriefing}
        disabled={loadingBriefing}
      >
        {loadingBriefing ? 'Generating...' : 'Get Morning Briefing'}
      </button>

      {#if error}
        <div class="briefing-error">{error}</div>
      {/if}

      {#if thinking}
        <details class="thinking-block">
          <summary>View AI reasoning</summary>
          <div class="thinking-content">{@html renderMarkdown(thinking)}</div>
        </details>
      {/if}

      {#if briefing}
        <div class="briefing-content">
          {@html renderMarkdown(briefing)}
        </div>
      {/if}

      <!-- Briefing history -->
      <button class="history-toggle" on:click={toggleHistory}>
        {showHistory ? 'Hide' : 'Show'} Past Briefings
      </button>

      {#if showHistory}
        {#if loadingHistory}
          <div class="loading-msg">Loading history...</div>
        {:else if briefingHistory.length === 0}
          <div class="empty-msg">No past briefings yet.</div>
        {:else}
          <div class="history-list">
            {#each briefingHistory as b}
              <details class="history-entry">
                <summary>
                  <span class="history-date">{formatDate(b.created_at)}</span>
                  <span class="history-regime">{b.market_regime}</span>
                </summary>
                <div class="history-tickers">
                  {#each b.tickers as t}
                    <div class="history-ticker">
                      <span class="ht-symbol">{t.ticker}</span>
                      <span class="ht-price">${t.price.toFixed(2)}</span>
                      <span class="ht-outlook" class:bullish={t.outlook === 'bullish'} class:bearish={t.outlook === 'bearish'}>
                        {t.outlook}
                      </span>
                    </div>
                  {/each}
                </div>
              </details>
            {/each}
          </div>
        {/if}
      {/if}
    </div>
  {/if}
</div>

<style>
  .watchlist {
    max-width: 600px;
    width: 100%;
    margin: 0 auto;
  }

  .watchlist-header {
    display: flex;
    justify-content: space-between;
    align-items: center;
    margin-bottom: 1rem;
  }

  .watchlist-header h2 {
    color: var(--text);
    font-size: 1.3rem;
    margin: 0;
  }

  .count {
    color: var(--text-muted);
    font-size: 0.8rem;
  }

  .add-row {
    display: flex;
    gap: 0.5rem;
    margin-bottom: 0.5rem;
  }

  .add-input {
    flex: 1;
    background: var(--bg-darker);
    border: 1px solid var(--border);
    color: var(--text);
    padding: 0.5rem 0.75rem;
    border-radius: 4px;
    font-family: inherit;
    font-size: 0.9rem;
    text-transform: uppercase;
  }

  .add-input::placeholder {
    color: var(--text-muted);
    text-transform: none;
  }

  .add-input:focus {
    outline: none;
    border-color: var(--accent);
  }

  .add-btn {
    background: var(--accent);
    color: var(--bg-dark);
    border: none;
    padding: 0.5rem 1rem;
    border-radius: 4px;
    font-family: inherit;
    font-weight: 600;
    cursor: pointer;
  }

  .add-btn:disabled {
    opacity: 0.5;
    cursor: not-allowed;
  }

  .add-error {
    color: #ef5350;
    font-size: 0.8rem;
    margin-bottom: 0.5rem;
  }

  .loading-msg,
  .empty-msg {
    color: var(--text-muted);
    text-align: center;
    padding: 2rem 0;
    font-size: 0.9rem;
  }

  .ticker-list {
    display: flex;
    flex-direction: column;
    gap: 0.25rem;
    margin-bottom: 1.5rem;
  }

  .ticker-row {
    display: flex;
    align-items: center;
    gap: 0.75rem;
    padding: 0.5rem 0.75rem;
    background: var(--bg-card);
    border: 1px solid var(--border);
    border-radius: 4px;
  }

  .ticker-info {
    display: flex;
    flex-direction: column;
    gap: 0.1rem;
    min-width: 80px;
  }

  .ticker-name {
    background: none;
    border: none;
    color: var(--accent);
    font-family: inherit;
    font-size: 0.95rem;
    font-weight: 600;
    cursor: pointer;
    padding: 0;
    text-align: left;
  }

  .ticker-name:hover {
    text-decoration: underline;
  }

  .company-name {
    color: var(--text-muted);
    font-size: 0.7rem;
    white-space: nowrap;
    overflow: hidden;
    text-overflow: ellipsis;
    max-width: 140px;
  }

  .ticker-price {
    flex: 1;
    display: flex;
    align-items: center;
    gap: 0.5rem;
    justify-content: flex-end;
  }

  .price {
    color: var(--text);
    font-size: 0.9rem;
  }

  .price.dim {
    color: var(--text-muted);
  }

  .change {
    font-size: 0.8rem;
    font-weight: 600;
  }

  .change.up {
    color: #26a69a;
  }

  .change.down {
    color: #ef5350;
  }

  .remove-btn {
    background: none;
    border: none;
    color: var(--text-muted);
    font-size: 0.9rem;
    cursor: pointer;
    padding: 0.2rem 0.4rem;
    line-height: 1;
    border-radius: 2px;
  }

  .remove-btn:hover {
    color: #ef5350;
    background: rgba(239, 83, 80, 0.1);
  }

  .briefing-section {
    border-top: 1px solid var(--border);
    padding-top: 1rem;
  }

  .briefing-btn {
    width: 100%;
    background: var(--bg-card);
    border: 1px solid var(--accent);
    color: var(--accent);
    padding: 0.6rem;
    border-radius: 4px;
    font-family: inherit;
    font-size: 0.9rem;
    cursor: pointer;
    transition: all 0.15s;
  }

  .briefing-btn:hover:not(:disabled) {
    background: var(--accent);
    color: var(--bg-dark);
  }

  .briefing-btn:disabled {
    opacity: 0.6;
    cursor: wait;
  }

  .briefing-error {
    color: #ef5350;
    font-size: 0.85rem;
    margin-top: 0.5rem;
  }

  .briefing-content {
    margin-top: 1rem;
    padding: 1rem;
    background: var(--bg-darker);
    border: 1px solid var(--border);
    border-radius: 4px;
    color: var(--text);
    font-size: 0.9rem;
    line-height: 1.6;
  }

  .briefing-content :global(p) {
    margin: 0 0 0.75rem 0;
  }

  .briefing-content :global(p:last-child) {
    margin-bottom: 0;
  }

  .briefing-content :global(strong) {
    color: var(--accent);
  }

  .thinking-block {
    margin-top: 0.75rem;
    border: 1px solid var(--border);
    border-radius: 4px;
    overflow: hidden;
  }

  .thinking-block summary {
    padding: 0.5rem 0.75rem;
    background: var(--bg-card);
    color: var(--text-muted);
    font-size: 0.8rem;
    cursor: pointer;
    user-select: none;
    transition: color 0.15s;
  }

  .thinking-block summary:hover {
    color: var(--text);
  }

  .thinking-content {
    padding: 0.75rem;
    background: var(--bg-darker);
    color: var(--text-muted);
    font-size: 0.8rem;
    line-height: 1.5;
    max-height: 300px;
    overflow-y: auto;
  }

  .thinking-content :global(p) {
    margin: 0 0 0.5rem 0;
  }

  .thinking-content :global(p:last-child) {
    margin-bottom: 0;
  }

  .history-toggle {
    width: 100%;
    background: none;
    border: 1px solid var(--border);
    color: var(--text-muted);
    padding: 0.4rem;
    border-radius: 4px;
    font-family: inherit;
    font-size: 0.8rem;
    cursor: pointer;
    margin-top: 0.75rem;
    transition: color 0.15s;
  }

  .history-toggle:hover {
    color: var(--text);
    border-color: var(--text-muted);
  }

  .history-list {
    display: flex;
    flex-direction: column;
    gap: 0.25rem;
    margin-top: 0.5rem;
  }

  .history-entry {
    border: 1px solid var(--border);
    border-radius: 4px;
    overflow: hidden;
  }

  .history-entry summary {
    padding: 0.5rem 0.75rem;
    background: var(--bg-card);
    cursor: pointer;
    display: flex;
    justify-content: space-between;
    align-items: center;
    font-size: 0.8rem;
  }

  .history-date {
    color: var(--text-muted);
  }

  .history-regime {
    color: var(--text);
    font-weight: 500;
    text-align: right;
    max-width: 60%;
    white-space: nowrap;
    overflow: hidden;
    text-overflow: ellipsis;
  }

  .history-tickers {
    padding: 0.5rem 0.75rem;
    background: var(--bg-darker);
    display: flex;
    flex-direction: column;
    gap: 0.25rem;
  }

  .history-ticker {
    display: flex;
    align-items: center;
    gap: 0.5rem;
    font-size: 0.8rem;
  }

  .ht-symbol {
    color: var(--accent);
    font-weight: 600;
    min-width: 45px;
  }

  .ht-price {
    color: var(--text);
  }

  .ht-outlook {
    margin-left: auto;
    font-size: 0.7rem;
    font-weight: 600;
    text-transform: uppercase;
    padding: 0.1rem 0.4rem;
    border-radius: 3px;
    background: rgba(255, 255, 255, 0.05);
    color: var(--text-muted);
  }

  .ht-outlook.bullish {
    color: #26a69a;
    background: rgba(38, 166, 154, 0.1);
  }

  .ht-outlook.bearish {
    color: #ef5350;
    background: rgba(239, 83, 80, 0.1);
  }
</style>
