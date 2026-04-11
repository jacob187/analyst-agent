<script lang="ts">
  import { createEventDispatcher, onMount } from 'svelte';

  const dispatch = createEventDispatcher<{
    open: { ticker: string; sessionId: string };
  }>();

  interface TickerEntry {
    ticker: string;
    session_count: number;
    last_active: string;
  }

  const API_BASE = import.meta.env.VITE_API_URL || 'http://localhost:8000';

  let companies: TickerEntry[] = [];
  let loading = true;
  let error = '';

  onMount(loadCompanies);

  async function loadCompanies() {
    loading = true;
    error = '';
    try {
      // GET /tickers returns ticker summaries without session IDs, preventing
      // enumeration — session IDs are only fetched when the user picks a company.
      const res = await fetch(`${API_BASE}/tickers`);
      if (!res.ok) throw new Error('Failed to load companies');
      const data = await res.json();
      companies = data.tickers || [];
    } catch (e) {
      error = e instanceof Error ? e.message : 'Failed to load companies';
    } finally {
      loading = false;
    }
  }

  async function openCompany(ticker: string) {
    // Fetch the latest session ID for this ticker on-demand so it is never
    // exposed in the public listing response.
    const res = await fetch(`${API_BASE}/sessions/by-ticker/${encodeURIComponent(ticker)}`);
    if (!res.ok) {
      error = `Failed to open ${ticker}`;
      return;
    }
    const data = await res.json();
    const sessionId: string | null = data.session?.id ?? null;
    dispatch('open', { ticker, sessionId: sessionId ?? '' });
  }

  function formatDate(dateStr: string): string {
    const date = new Date(dateStr);
    const now = new Date();
    const diff = now.getTime() - date.getTime();
    const days = Math.floor(diff / (1000 * 60 * 60 * 24));
    if (days === 0) return 'Today';
    if (days === 1) return 'Yesterday';
    if (days < 7) return date.toLocaleDateString([], { weekday: 'short' });
    return date.toLocaleDateString([], { month: 'short', day: 'numeric' });
  }
</script>

<div class="companies-page">
  <div class="page-header">
    <h2>Saved Companies</h2>
    <p>Companies with saved chat sessions. Click to open the dashboard.</p>
  </div>

  {#if loading}
    <div class="state-msg">Loading...</div>
  {:else if error}
    <div class="state-msg error">{error}</div>
  {:else if companies.length === 0}
    <div class="state-msg">
      No saved companies yet. Enter a ticker to start a chat.
    </div>
  {:else}
    <div class="grid">
      {#each companies as company}
        <!-- svelte-ignore a11y-click-events-have-key-events -->
        <!-- svelte-ignore a11y-no-static-element-interactions -->
        <div
          class="card"
          on:click={() => openCompany(company.ticker)}
        >
          <div class="card-top">
            <span class="ticker">{company.ticker}</span>
            <span class="badge">{company.session_count} {company.session_count === 1 ? 'session' : 'sessions'}</span>
          </div>
          <div class="card-bottom">
            <span class="last-active">Last active: {formatDate(company.last_active)}</span>
            <span class="arrow">→</span>
          </div>
        </div>
      {/each}
    </div>
  {/if}
</div>

<style>
  .companies-page {
    flex: 1;
    overflow-y: auto;
    padding: 1.5rem 0;
    min-height: 0;
  }

  .page-header {
    margin-bottom: 2rem;
  }

  .page-header h2 {
    color: var(--text);
    font-size: 1.3rem;
    font-weight: 600;
    margin: 0 0 0.4rem;
  }

  .page-header p {
    color: var(--text-dim);
    font-size: 0.88rem;
    margin: 0;
  }

  .state-msg {
    color: var(--text-dim);
    font-size: 0.9rem;
    padding: 3rem 0;
    text-align: center;
  }

  .state-msg.error {
    color: var(--danger, #ff4d4d);
  }

  .grid {
    display: grid;
    grid-template-columns: repeat(auto-fill, minmax(200px, 1fr));
    gap: 1rem;
  }

  .card {
    background: var(--bg-card);
    border: 1px solid var(--border);
    border-radius: 6px;
    padding: 1.2rem 1.4rem;
    cursor: pointer;
    transition: border-color 0.2s, background 0.2s;
    display: flex;
    flex-direction: column;
    gap: 0.8rem;
  }

  .card:hover {
    border-color: var(--accent);
    background: rgba(0, 255, 157, 0.03);
  }

  .card-top {
    display: flex;
    align-items: center;
    justify-content: space-between;
    gap: 0.5rem;
  }

  .ticker {
    font-size: 1.15rem;
    font-weight: 700;
    color: var(--text);
    letter-spacing: 0.05em;
  }

  .badge {
    font-size: 0.7rem;
    color: var(--text-muted);
    background: var(--bg-darker, rgba(255,255,255,0.06));
    border-radius: 3px;
    padding: 0.15rem 0.4rem;
    white-space: nowrap;
  }

  .card-bottom {
    display: flex;
    align-items: center;
    justify-content: space-between;
  }

  .last-active {
    font-size: 0.78rem;
    color: var(--text-muted);
  }

  .arrow {
    color: var(--accent);
    font-size: 0.9rem;
    opacity: 0;
    transition: opacity 0.2s;
  }

  .card:hover .arrow {
    opacity: 1;
  }

  @media (max-width: 480px) {
    .grid {
      grid-template-columns: 1fr 1fr;
      gap: 0.75rem;
    }

    .card {
      padding: 1rem;
    }
  }
</style>
