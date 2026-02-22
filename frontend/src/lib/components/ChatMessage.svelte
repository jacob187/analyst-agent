<script lang="ts">
  import { marked } from 'marked';

  export let role: 'user' | 'assistant';
  export let content: string;
  export let thinking: string = '';
  export let status: string = '';

  // Convert markdown to HTML — reactive, re-runs when content changes
  $: htmlContent = marked.parse(content);
  $: thinkingHtml = thinking ? marked.parse(thinking) : '';
</script>

<div class="message {role}">
  <div class="header">
    <span class="label">{role === 'user' ? 'YOU' : 'ANALYST'}</span>
    {#if status}
      <span class="status-badge">{status}</span>
    {/if}
  </div>

  <!-- Thinking block: collapsible <details> so users can peek at reasoning -->
  {#if thinking}
    <details class="thinking-block">
      <summary>Thinking</summary>
      <div class="thinking-content">{@html thinkingHtml}</div>
    </details>
  {/if}

  {#if content}
    <div class="content">{@html htmlContent}</div>
  {/if}
</div>

<style>
  .message {
    margin-bottom: 1.2rem;
    animation: slideIn 0.2s ease-out;
  }

  @keyframes slideIn {
    from {
      opacity: 0;
      transform: translateY(10px);
    }
    to {
      opacity: 1;
      transform: translateY(0);
    }
  }

  .header {
    margin-bottom: 0.5rem;
  }

  .label {
    font-size: 0.7rem;
    font-weight: 600;
    letter-spacing: 0.1em;
    opacity: 0.6;
  }

  .message.user .label {
    color: var(--accent);
  }

  .message.assistant .label {
    color: var(--success);
  }

  .status-badge {
    font-size: 0.65rem;
    color: var(--accent);
    opacity: 0.8;
    font-style: italic;
  }

  .thinking-block {
    margin-bottom: 0.5rem;
    border: 1px solid rgba(255, 255, 255, 0.1);
    border-radius: 4px;
    overflow: hidden;
  }

  .thinking-block summary {
    padding: 0.4rem 0.8rem;
    font-size: 0.75rem;
    font-weight: 600;
    letter-spacing: 0.05em;
    color: var(--text-muted);
    cursor: pointer;
    background: rgba(255, 255, 255, 0.03);
    user-select: none;
  }

  .thinking-block summary:hover {
    color: var(--text);
    background: rgba(255, 255, 255, 0.06);
  }

  .thinking-content {
    padding: 0.6rem 0.8rem;
    font-size: 0.8rem;
    line-height: 1.4;
    color: var(--text-muted);
    font-family: 'Monaco', 'Menlo', 'Courier New', monospace;
    background: rgba(0, 0, 0, 0.2);
    white-space: pre-wrap;
    word-wrap: break-word;
    max-height: 300px;
    overflow-y: auto;
  }

  .content {
    padding: 0.8rem 1rem;
    border-radius: 4px;
    line-height: 1.5;
    white-space: pre-wrap;
    word-wrap: break-word;
  }

  .message.user .content {
    background: var(--bg-input);
    border-left: 2px solid var(--accent);
  }

  .message.assistant .content {
    background: var(--bg-card);
    border-left: 2px solid var(--success);
  }

  /* Markdown element styling */
  /* :global() tells Svelte these styles apply to dynamically rendered HTML */
  .content :global(h1),
  .content :global(h2),
  .content :global(h3),
  .content :global(h4),
  .content :global(h5),
  .content :global(h6) {
    margin: 0.8em 0 0.4em;
    font-weight: 600;
    line-height: 1.3;
  }

  .content :global(h1) { font-size: 1.5em; }
  .content :global(h2) { font-size: 1.3em; }
  .content :global(h3) { font-size: 1.1em; }

  .content :global(p) {
    margin: 0.5em 0;
  }

  .content :global(p:first-child) {
    margin-top: 0;
  }

  .content :global(p:last-child) {
    margin-bottom: 0;
  }

  /* Inline code */
  .content :global(code) {
    background: rgba(255, 255, 255, 0.1);
    padding: 0.2em 0.4em;
    border-radius: 3px;
    font-family: 'Monaco', 'Menlo', 'Courier New', monospace;
    font-size: 0.9em;
  }

  /* Code blocks */
  .content :global(pre) {
    background: rgba(0, 0, 0, 0.3);
    padding: 1em;
    border-radius: 4px;
    overflow-x: auto;
    margin: 0.8em 0;
  }

  .content :global(pre code) {
    background: none;
    padding: 0;
    font-size: 0.85em;
  }

  /* Lists */
  .content :global(ul),
  .content :global(ol) {
    margin: 0.5em 0;
    padding-left: 1.8em;
  }

  .content :global(li) {
    margin: 0.3em 0;
  }

  /* Links */
  .content :global(a) {
    color: var(--accent);
    text-decoration: underline;
  }

  .content :global(a:hover) {
    opacity: 0.8;
  }

  /* Blockquotes */
  .content :global(blockquote) {
    border-left: 3px solid var(--accent);
    padding-left: 1em;
    margin: 0.8em 0;
    opacity: 0.8;
  }

  /* Horizontal rules */
  .content :global(hr) {
    border: none;
    border-top: 1px solid rgba(255, 255, 255, 0.2);
    margin: 1em 0;
  }

  /* Tables */
  .content :global(table) {
    border-collapse: collapse;
    width: 100%;
    margin: 0.8em 0;
  }

  .content :global(th),
  .content :global(td) {
    border: 1px solid rgba(255, 255, 255, 0.2);
    padding: 0.5em;
    text-align: left;
  }

  .content :global(th) {
    background: rgba(255, 255, 255, 0.1);
    font-weight: 600;
  }

  /* Tablet breakpoint */
  @media (max-width: 768px) {
    .message {
      margin-bottom: 1rem;
    }

    .content {
      padding: 0.7rem 0.9rem;
      font-size: 0.9rem;
    }
  }

  /* Small mobile breakpoint */
  @media (max-width: 480px) {
    .message {
      margin-bottom: 0.8rem;
    }

    .label {
      font-size: 0.65rem;
    }

    .content {
      padding: 0.6rem 0.8rem;
      font-size: 0.85rem;
      line-height: 1.4;
    }
  }
</style>
