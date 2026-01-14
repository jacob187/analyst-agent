<script lang="ts">
  import { onMount } from 'svelte';
  import ChatMessage from './ChatMessage.svelte';

  export let sessionId: string;
  export let ticker: string;

  interface Message {
    role: 'user' | 'assistant';
    content: string;
    created_at: string;
  }

  let messages: Message[] = [];
  let loading = true;
  let error = '';

  onMount(async () => {
    await loadMessages();
  });

  async function loadMessages() {
    loading = true;
    error = '';
    try {
      const apiHost = import.meta.env.VITE_API_URL || 'localhost:8000';
      const response = await fetch(`http://${apiHost}/sessions/${sessionId}/messages`);
      if (!response.ok) throw new Error('Failed to load messages');
      const data = await response.json();
      messages = data.messages;
    } catch (e) {
      error = e instanceof Error ? e.message : 'Failed to load messages';
    } finally {
      loading = false;
    }
  }
</script>

<div class="chat-viewer">
  <div class="header">
    <div class="ticker-display">
      <span class="symbol">{ticker}</span>
      <span class="status archived">ARCHIVED</span>
    </div>
  </div>

  <div class="messages">
    {#if loading}
      <div class="loading">Loading messages...</div>
    {:else if error}
      <div class="error">{error}</div>
    {:else if messages.length === 0}
      <div class="empty">No messages in this session</div>
    {:else}
      {#each messages as message}
        <ChatMessage role={message.role} content={message.content} />
      {/each}
    {/if}
  </div>

  <div class="footer">
    <span class="read-only-notice">Read-only view of past conversation</span>
  </div>
</div>

<style>
  .chat-viewer {
    display: flex;
    flex-direction: column;
    height: 100%;
    background: var(--bg-darker);
    border: 1px solid var(--border);
    border-radius: 8px;
    overflow: hidden;
  }

  .header {
    padding: 1rem 1.5rem;
    background: var(--bg-card);
    border-bottom: 1px solid var(--border);
  }

  .ticker-display {
    display: flex;
    align-items: center;
    gap: 1rem;
  }

  .symbol {
    font-size: 1.5rem;
    font-weight: 700;
    letter-spacing: 0.05em;
    color: var(--text);
  }

  .status {
    padding: 0.2rem 0.6rem;
    border-radius: 3px;
    font-size: 0.7rem;
    font-weight: 600;
    letter-spacing: 0.1em;
  }

  .status.archived {
    background: rgba(128, 128, 128, 0.2);
    color: var(--text-dim);
  }

  .messages {
    flex: 1;
    padding: 1.5rem;
    overflow-y: auto;
    min-height: 400px;
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

  .footer {
    padding: 1rem 1.5rem;
    background: var(--bg-card);
    border-top: 1px solid var(--border);
    text-align: center;
  }

  .read-only-notice {
    color: var(--text-muted);
    font-size: 0.8rem;
    font-style: italic;
  }

  @media (max-width: 768px) {
    .header {
      padding: 0.8rem 1rem;
    }

    .symbol {
      font-size: 1.25rem;
    }

    .messages {
      padding: 1rem;
      min-height: 300px;
    }

    .footer {
      padding: 0.8rem 1rem;
    }
  }

  @media (max-width: 480px) {
    .header {
      padding: 0.6rem 0.8rem;
    }

    .ticker-display {
      gap: 0.6rem;
    }

    .symbol {
      font-size: 1.1rem;
    }

    .status {
      padding: 0.15rem 0.5rem;
      font-size: 0.65rem;
    }

    .messages {
      padding: 0.8rem;
      min-height: 250px;
    }

    .footer {
      padding: 0.6rem 0.8rem;
    }

    .read-only-notice {
      font-size: 0.75rem;
    }
  }
</style>
