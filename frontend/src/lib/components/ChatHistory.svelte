<script lang="ts">
  import { createEventDispatcher, onMount } from 'svelte';

  const dispatch = createEventDispatcher<{
    select: { sessionId: string; ticker: string };
    continue: { sessionId: string; ticker: string };
    close: void;
  }>();

  interface Session {
    id: string;
    ticker: string;
    created_at: string;
  }

  let sessions: Session[] = [];
  let loading = true;
  let error = '';

  onMount(async () => {
    await loadSessions();
  });

  async function loadSessions() {
    loading = true;
    error = '';
    try {
      const apiHost = import.meta.env.VITE_API_URL || 'localhost:8000';
      const response = await fetch(`http://${apiHost}/sessions`);
      if (!response.ok) throw new Error('Failed to load sessions');
      const data = await response.json();
      sessions = data.sessions;
    } catch (e) {
      error = e instanceof Error ? e.message : 'Failed to load chat history';
    } finally {
      loading = false;
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
    {:else if sessions.length === 0}
      <div class="empty">No chat history yet</div>
    {:else}
      {#each sessions as session}
        <div class="session-item">
          <div class="session-info">
            <span class="ticker">{session.ticker}</span>
            <span class="date">{formatDate(session.created_at)}</span>
          </div>
          <div class="session-actions">
            <button class="btn btn-sm btn-ghost" on:click={() => viewSession(session)}>View</button>
            <button class="btn btn-sm btn-primary" on:click={() => continueSession(session)}>Continue</button>
          </div>
        </div>
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
