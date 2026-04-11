<script lang="ts">
  import { createEventDispatcher, onMount } from 'svelte';

  const dispatch = createEventDispatcher<{
    select: { sessionId: string; ticker: string };
    continue: { sessionId: string; ticker: string };
    close: void;
  }>();

  interface TickerEntry {
    ticker: string;
    session_count: number;
    last_active: string;
  }

  interface Session {
    id: string;
    ticker: string;
    created_at: string;
    model: string | null;
  }

  const apiBase = import.meta.env.VITE_API_URL || 'http://localhost:8000';

  // First level: tickers (no session IDs — safe to list publicly)
  let tickers: TickerEntry[] = [];
  // Second level: sessions for the expanded ticker
  let expandedTicker: string | null = null;
  let tickerSessions: Session[] = [];
  let tickerSessionsLoading = false;

  let loading = true;
  let error = '';

  onMount(async () => {
    await loadTickers();
  });

  async function loadTickers() {
    loading = true;
    error = '';
    try {
      // GET /tickers returns ticker summaries without session IDs, so the
      // listing is safe — IDs are only fetched when the user drills into a ticker.
      const response = await fetch(`${apiBase}/tickers`);
      if (!response.ok) throw new Error('Failed to load chat history');
      const data = await response.json();
      tickers = data.tickers;
    } catch (e) {
      error = e instanceof Error ? e.message : 'Failed to load chat history';
    } finally {
      loading = false;
    }
  }

  async function toggleTicker(ticker: string) {
    if (expandedTicker === ticker) {
      expandedTicker = null;
      tickerSessions = [];
      return;
    }
    expandedTicker = ticker;
    tickerSessionsLoading = true;
    try {
      const response = await fetch(`${apiBase}/sessions?ticker=${encodeURIComponent(ticker)}`);
      if (!response.ok) throw new Error('Failed to load sessions');
      const data = await response.json();
      tickerSessions = data.sessions;
    } catch (e) {
      error = e instanceof Error ? e.message : 'Failed to load sessions';
    } finally {
      tickerSessionsLoading = false;
    }
  }

  function formatDate(dateStr: string): string {
    const date = new Date(dateStr);
    const now = new Date();
    const diff = now.getTime() - date.getTime();
    const days = Math.floor(diff / (1000 * 60 * 60 * 24));

    if (days === 0) {
      return date.toLocaleTimeString([], { hour: '2-digit', minute: '2-digit' });
    } else if (days === 1) {
      return 'Yesterday';
    } else if (days < 7) {
      return date.toLocaleDateString([], { weekday: 'short' });
    } else {
      return date.toLocaleDateString([], { month: 'short', day: 'numeric' });
    }
  }

  function viewSession(session: Session) {
    dispatch('select', { sessionId: session.id, ticker: session.ticker });
  }

  function continueSession(session: Session) {
    dispatch('continue', { sessionId: session.id, ticker: session.ticker });
  }

  async function deleteSession(session: Session) {
    if (!confirm(`Delete chat with ${session.ticker}?`)) return;
    try {
      const response = await fetch(
        `${apiBase}/sessions/${session.id}?ticker=${encodeURIComponent(session.ticker)}`,
        { method: 'DELETE' }
      );
      if (!response.ok) throw new Error('Failed to delete session');
      tickerSessions = tickerSessions.filter(s => s.id !== session.id);
      // If that was the last session for this ticker, collapse and reload tickers
      if (tickerSessions.length === 0) {
        expandedTicker = null;
        await loadTickers();
      }
    } catch (e) {
      error = e instanceof Error ? e.message : 'Failed to delete session';
    }
  }
</script>

<div class="chat-history">
  <div class="header">
    <h3>Chat History</h3>
    <button class="close-btn" on:click={() => dispatch('close')}>x</button>
  </div>

  <div class="sessions-list">
    {#if loading}
      <div class="loading">Loading...</div>
    {:else if error}
      <div class="error">{error}</div>
    {:else if tickers.length === 0}
      <div class="empty">No chat history yet</div>
    {:else}
      {#each tickers as entry}
        <!-- Ticker row — click to expand/collapse its sessions -->
        <button class="ticker-row" on:click={() => toggleTicker(entry.ticker)}>
          <div class="ticker-row-info">
            <span class="ticker">{entry.ticker}</span>
            <span class="model-badge">{entry.session_count} {entry.session_count === 1 ? 'session' : 'sessions'}</span>
          </div>
          <div class="ticker-row-right">
            <span class="date">{formatDate(entry.last_active)}</span>
            <span class="chevron">{expandedTicker === entry.ticker ? '▲' : '▼'}</span>
          </div>
        </button>

        {#if expandedTicker === entry.ticker}
          <div class="sessions-nested">
            {#if tickerSessionsLoading}
              <div class="loading-inner">Loading...</div>
            {:else}
              {#each tickerSessions as session}
                <div class="session-item">
                  <div class="session-info">
                    {#if session.model}
                      <span class="model-badge">{session.model}</span>
                    {/if}
                    <span class="date">{formatDate(session.created_at)}</span>
                  </div>
                  <div class="session-actions">
                    <button class="btn btn-sm btn-ghost" on:click={() => viewSession(session)}>View</button>
                    <button class="btn btn-sm btn-primary" on:click={() => continueSession(session)}>Continue</button>
                    <button class="btn btn-sm btn-danger" on:click={() => deleteSession(session)}>Delete</button>
                  </div>
                </div>
              {/each}
            {/if}
          </div>
        {/if}
      {/each}
    {/if}
  </div>
</div>

<style>
  .chat-history {
    background: var(--bg-card);
    border: 1px solid var(--border);
    border-radius: 8px;
    max-width: 400px;
    width: 100%;
    max-height: 500px;
    display: flex;
    flex-direction: column;
  }

  .header {
    display: flex;
    justify-content: space-between;
    align-items: center;
    padding: 1rem 1.5rem;
    border-bottom: 1px solid var(--border);
  }

  .header h3 {
    margin: 0;
    font-size: 1rem;
    color: var(--text);
    font-weight: 600;
  }

  .close-btn {
    background: none;
    border: none;
    color: var(--text-dim);
    font-size: 1.2rem;
    cursor: pointer;
    padding: 0.2rem 0.5rem;
    transition: color 0.2s;
  }

  .close-btn:hover {
    color: var(--accent);
  }

  .sessions-list {
    flex: 1;
    overflow-y: auto;
    padding: 0.5rem;
  }

  .loading, .error, .empty {
    padding: 2rem;
    text-align: center;
    color: var(--text-dim);
    font-size: 0.9rem;
  }

  .error {
    color: var(--danger);
  }

  .session-item {
    display: flex;
    justify-content: space-between;
    align-items: center;
    width: 100%;
    padding: 0.8rem 1rem;
    background: transparent;
    border: 1px solid transparent;
    border-radius: 4px;
    transition: all 0.2s;
  }

  .session-item:hover {
    background: var(--bg-darker);
    border-color: var(--border);
  }

  .session-info {
    display: flex;
    flex-direction: column;
    gap: 0.2rem;
  }

  .session-actions {
    display: flex;
    gap: 0.5rem;
  }

  .ticker {
    font-weight: 600;
    color: var(--text);
    font-size: 0.95rem;
    letter-spacing: 0.02em;
  }

  .date {
    color: var(--text-muted);
    font-size: 0.8rem;
  }

  .model-badge {
    display: inline-block;
    font-size: 0.7rem;
    color: var(--text-muted);
    background: var(--bg-darker);
    border-radius: 3px;
    padding: 0.1rem 0.4rem;
    letter-spacing: 0.01em;
  }

  /* Ticker accordion row */
  .ticker-row {
    display: flex;
    justify-content: space-between;
    align-items: center;
    width: 100%;
    padding: 0.7rem 1rem;
    background: transparent;
    border: none;
    border-radius: 4px;
    cursor: pointer;
    transition: background 0.15s;
    text-align: left;
  }

  .ticker-row:hover {
    background: var(--bg-darker);
  }

  .ticker-row-info {
    display: flex;
    align-items: center;
    gap: 0.5rem;
  }

  .ticker-row-right {
    display: flex;
    align-items: center;
    gap: 0.5rem;
    color: var(--text-muted);
    font-size: 0.8rem;
  }

  .chevron {
    font-size: 0.6rem;
    color: var(--accent);
  }

  /* Nested sessions under an expanded ticker */
  .sessions-nested {
    margin: 0 0 0.3rem 1rem;
    border-left: 2px solid var(--border);
    padding-left: 0.5rem;
  }

  .loading-inner {
    padding: 0.5rem 1rem;
    font-size: 0.8rem;
    color: var(--text-muted);
  }

  @media (max-width: 480px) {
    .chat-history {
      max-width: 100%;
      max-height: 400px;
    }

    .header {
      padding: 0.8rem 1rem;
    }

    .session-item {
      padding: 0.7rem 0.8rem;
    }
  }
</style>
