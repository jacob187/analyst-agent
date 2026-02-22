<script lang="ts">
  import { onMount, afterUpdate } from 'svelte';
  import ChatMessage from './ChatMessage.svelte';

  export let ticker: string;
  export let googleApiKey: string;
  export let secHeader: string;
  export let tavilyApiKey: string = '';
  export let sessionId: string | null = null;  // For continuing existing sessions

  interface Message {
    role: 'user' | 'assistant';
    content: string;
    thinking?: string;
    status?: string;
    isStreaming?: boolean;
  }

  let messages: Message[] = [];
  let input = '';
  let socket: WebSocket | null = null;
  let connected = false;
  let authenticated = false;
  let messagesContainer: HTMLElement;
  let reconnectTimeout: ReturnType<typeof setTimeout> | null = null;

  onMount(async () => {
    // If continuing a session, load existing messages first
    if (sessionId) {
      await loadExistingMessages();
    }
    connectWebSocket();

    // Handle page visibility changes - reconnect if needed when user returns
    function handleVisibilityChange() {
      if (document.visibilityState === 'visible') {
        checkAndReconnect();
      }
    }

    // Handle window focus - reconnect if needed when user clicks back into the window
    function handleWindowFocus() {
      checkAndReconnect();
    }

    document.addEventListener('visibilitychange', handleVisibilityChange);
    window.addEventListener('focus', handleWindowFocus);

    return () => {
      document.removeEventListener('visibilitychange', handleVisibilityChange);
      window.removeEventListener('focus', handleWindowFocus);
      if (reconnectTimeout) clearTimeout(reconnectTimeout);
      socket?.close();
    };
  });

  async function loadExistingMessages() {
    try {
      const apiHost = import.meta.env.VITE_API_URL || 'localhost:8000';
      const response = await fetch(`http://${apiHost}/sessions/${sessionId}/messages`);
      if (response.ok) {
        const data = await response.json();
        messages = data.messages.map((m: { role: string; content: string }) => ({
          role: m.role as 'user' | 'assistant',
          content: m.content
        }));
      }
    } catch (e) {
      console.error('Failed to load existing messages:', e);
    }
  }

  afterUpdate(() => {
    if (messagesContainer) {
      messagesContainer.scrollTop = messagesContainer.scrollHeight;
    }
  });

  function checkAndReconnect() {
    if (!socket || socket.readyState === WebSocket.CLOSED || socket.readyState === WebSocket.CLOSING) {
      connected = false;
      authenticated = false;
      connectWebSocket(true);
    }
  }

  function connectWebSocket(isReconnect = false) {
    const apiHost = import.meta.env.VITE_API_URL || 'localhost:8000';
    const wsProtocol = window.location.protocol === 'https:' ? 'wss:' : 'ws:';
    const wsUrl = `${wsProtocol}//${apiHost}/ws/chat/${ticker}`;
    socket = new WebSocket(wsUrl);

    if (isReconnect) {
      messages = [...messages, {
        role: 'assistant',
        content: '[SYSTEM] Reconnecting...'
      }];
    }

    socket.onopen = () => {
      connected = true;
      // Send authentication message first
      const authPayload: Record<string, string> = {
        type: 'auth',
        google_api_key: googleApiKey,
        sec_header: secHeader
      };
      if (tavilyApiKey.trim()) {
        authPayload.tavily_api_key = tavilyApiKey;
      }
      if (sessionId) {
        authPayload.session_id = sessionId;
      }
      socket?.send(JSON.stringify(authPayload));
    };

    socket.onmessage = (event) => {
      const data = JSON.parse(event.data);

      if (data.type === 'auth_success') {
        authenticated = true;
        if (!sessionId) {
          messages = [{
            role: 'assistant',
            content: data.message
          }];
        } else {
          messages = [...messages, {
            role: 'assistant',
            content: '[SESSION RESUMED] ' + data.message
          }];
        }
      } else if (data.type === 'system') {
        messages = [...messages, {
          role: 'assistant',
          content: `[SYSTEM] ${data.message}`
        }];

      // --- Streaming event types ---
      // These all update a single "streaming" message in-place rather
      // than appending new messages for each event.

      } else if (data.type === 'node') {
        // Graph node transition: update the streaming message's status line
        updateStreamingMessage({ status: data.message });

      } else if (data.type === 'tool') {
        // Tool execution: show tool name + step progress
        const toolStatus = `Running ${data.tool} (step ${data.step}/${data.total})`;
        updateStreamingMessage({ status: toolStatus });

      } else if (data.type === 'thinking') {
        // Thinking token from the LLM: append to thinking field
        appendToStreamingMessage('thinking', data.message);

      } else if (data.type === 'token') {
        // Content token from the LLM: append to content field
        appendToStreamingMessage('content', data.message);

      } else if (data.type === 'response') {
        // Final assembled response: finalize the streaming message
        finalizeStreamingMessage(data.message);

      } else if (data.type === 'error') {
        messages = [...messages, {
          role: 'assistant',
          content: `[ERROR] ${data.message}`
        }];
      } else {
        messages = [...messages, {
          role: 'assistant',
          content: data.message || JSON.stringify(data)
        }];
      }
    };

    socket.onerror = () => {
      connected = false;
      authenticated = false;
      messages = [...messages, {
        role: 'assistant',
        content: '[ERROR] Connection error. Is the API running?'
      }];
    };

    socket.onclose = () => {
      connected = false;
      authenticated = false;
      // Auto-reconnect after a short delay if the page is visible
      if (document.visibilityState === 'visible') {
        reconnectTimeout = setTimeout(() => {
          connectWebSocket(true);
        }, 2000);
      }
    };
  }

  function sendMessage() {
    if (!input.trim() || !socket || !connected || !authenticated) return;

    // Add user message + a blank streaming message for the assistant response
    messages = [
      ...messages,
      { role: 'user', content: input },
      { role: 'assistant', content: '', thinking: '', status: 'Processing query...', isStreaming: true },
    ];

    socket.send(JSON.stringify({
      type: 'query',
      message: input
    }));
    input = '';
  }

  /**
   * Find (or create) the current streaming message and set fields on it.
   * This is how node/tool events update the status line in-place.
   */
  function updateStreamingMessage(fields: Partial<Message>) {
    const idx = messages.findIndex(m => m.isStreaming);
    if (idx === -1) return;
    messages[idx] = { ...messages[idx], ...fields };
    messages = messages; // trigger Svelte reactivity
  }

  /**
   * Append text to a field on the streaming message (content or thinking).
   * Tokens arrive one at a time, so we concat them as they come in.
   */
  function appendToStreamingMessage(field: 'content' | 'thinking', text: string) {
    const idx = messages.findIndex(m => m.isStreaming);
    if (idx === -1) return;
    const current = (messages[idx][field] || '') as string;
    messages[idx] = { ...messages[idx], [field]: current + text };
    messages = messages;
  }

  /**
   * Replace the streaming message with the final assembled response.
   * Clears status and the isStreaming flag so the message renders normally.
   */
  function finalizeStreamingMessage(fullContent: string) {
    const idx = messages.findIndex(m => m.isStreaming);
    if (idx === -1) {
      // Fallback: no streaming message found, just append
      messages = [...messages, { role: 'assistant', content: fullContent }];
      return;
    }
    messages[idx] = {
      ...messages[idx],
      content: fullContent,
      status: '',
      isStreaming: false,
    };
    messages = messages;
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
      <ChatMessage
        role={message.role}
        content={message.content}
        thinking={message.thinking || ''}
        status={message.status || ''}
      />
    {/each}
  </div>

  <div class="input-area">
    <textarea
      bind:value={input}
      on:keydown={handleKeydown}
      placeholder="ask anything..."
      disabled={!connected}
      rows="3"
    ></textarea>
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

  /* Tablet breakpoint */
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

    .input-area {
      padding: 0.8rem 1rem;
      flex-direction: column;
    }

    textarea {
      font-size: 16px; /* Prevents iOS zoom on focus */
    }

    button {
      width: 100%;
      padding: 0.9rem 1.5rem;
    }
  }

  /* Small mobile breakpoint */
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

    .input-area {
      padding: 0.6rem 0.8rem;
      gap: 0.6rem;
    }

    textarea {
      padding: 0.7rem;
      rows: 2;
    }

    button {
      padding: 0.8rem 1rem;
      font-size: 0.8rem;
    }
  }
</style>
