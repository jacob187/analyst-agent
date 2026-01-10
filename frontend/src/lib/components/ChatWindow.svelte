<script lang="ts">
  import { onMount, afterUpdate } from 'svelte';
  import ChatMessage from './ChatMessage.svelte';

  export let ticker: string;

  interface Message {
    role: 'user' | 'assistant';
    content: string;
  }

  let messages: Message[] = [];
  let input = '';
  let socket: WebSocket | null = null;
  let connected = false;
  let messagesContainer: HTMLElement;

  onMount(() => {
    connectWebSocket();
    return () => socket?.close();
  });

  afterUpdate(() => {
    if (messagesContainer) {
      messagesContainer.scrollTop = messagesContainer.scrollHeight;
    }
  });

  function connectWebSocket() {
    const apiHost = import.meta.env.VITE_API_URL || 'localhost:8000';
    const wsProtocol = window.location.protocol === 'https:' ? 'wss:' : 'ws:';
    const wsUrl = `${wsProtocol}//${apiHost}/ws/chat/${ticker}`;
    socket = new WebSocket(wsUrl);

    socket.onopen = () => {
      connected = true;
      messages = [{
        role: 'assistant',
        content: `Connected. Ready to analyze ${ticker}.`
      }];
    };

    socket.onmessage = (event) => {
      const data = JSON.parse(event.data);
      messages = [...messages, {
        role: 'assistant',
        content: data.message
      }];
    };

    socket.onerror = () => {
      connected = false;
      messages = [...messages, {
        role: 'assistant',
        content: 'Connection error. Is the API running?'
      }];
    };

    socket.onclose = () => {
      connected = false;
    };
  }

  function sendMessage() {
    if (!input.trim() || !socket || !connected) return;

    messages = [...messages, {
      role: 'user',
      content: input
    }];

    socket.send(input);
    input = '';
  }

  function handleKeydown(e: KeyboardEvent) {
    if (e.key === 'Enter' && !e.shiftKey) {
      e.preventDefault();
      sendMessage();
    }
  }
</script>

<div class="chat-window">
  <div class="header">
    <div class="ticker-display">
      <span class="symbol">{ticker}</span>
      <span class="status {connected ? 'online' : 'offline'}">
        {connected ? 'LIVE' : 'OFFLINE'}
      </span>
    </div>
  </div>

  <div class="messages" bind:this={messagesContainer}>
    {#each messages as message}
      <ChatMessage role={message.role} content={message.content} />
    {/each}
  </div>

  <div class="input-area">
    <textarea
      bind:value={input}
      on:keydown={handleKeydown}
      placeholder="ask anything..."
      disabled={!connected}
      rows="3"
    />
    <button on:click={sendMessage} disabled={!connected || !input.trim()}>
      send
    </button>
  </div>
</div>

<style>
  .chat-window {
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

  .status.online {
    background: rgba(0, 255, 157, 0.15);
    color: var(--success);
  }

  .status.offline {
    background: rgba(255, 51, 102, 0.15);
    color: var(--danger);
  }

  .messages {
    flex: 1;
    padding: 1.5rem;
    overflow-y: auto;
    min-height: 400px;
  }

  .input-area {
    padding: 1rem 1.5rem;
    background: var(--bg-card);
    border-top: 1px solid var(--border);
    display: flex;
    gap: 0.8rem;
  }

  textarea {
    flex: 1;
    background: var(--bg-input);
    border: 1px solid var(--border);
    border-radius: 4px;
    padding: 0.8rem;
    color: var(--text);
    font-family: inherit;
    font-size: 0.9rem;
    resize: none;
    transition: border-color 0.2s;
  }

  textarea:focus {
    outline: none;
    border-color: var(--accent);
  }

  textarea::placeholder {
    color: var(--text-muted);
  }

  textarea:disabled {
    opacity: 0.5;
    cursor: not-allowed;
  }

  button {
    background: var(--accent);
    color: var(--bg-dark);
    border: none;
    padding: 0.8rem 1.5rem;
    border-radius: 4px;
    font-size: 0.85rem;
    font-weight: 600;
    cursor: pointer;
    transition: all 0.2s;
    text-transform: uppercase;
    letter-spacing: 0.05em;
    align-self: flex-end;
  }

  button:hover:not(:disabled) {
    background: var(--accent-dim);
    transform: translateY(-1px);
  }

  button:disabled {
    opacity: 0.4;
    cursor: not-allowed;
  }
</style>
