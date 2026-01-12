<script lang="ts">
  import { createEventDispatcher } from 'svelte';

  export let disabled = false;

  let ticker = '';
  const dispatch = createEventDispatcher();

  function handleSubmit() {
    if (ticker.trim() && !disabled) {
      dispatch('submit', ticker.toUpperCase());
      ticker = '';
    }
  }

  function handleKeydown(e: KeyboardEvent) {
    if (e.key === 'Enter' && !e.shiftKey) {
      e.preventDefault();
      handleSubmit();
    }
  }
</script>

<div class="ticker-input">
  <span class="prompt">$</span>
  <input
    type="text"
    bind:value={ticker}
    on:keydown={handleKeydown}
    placeholder="enter ticker"
    {disabled}
    maxlength="5"
  />
  <button on:click={handleSubmit} {disabled}>
    analyze
  </button>
</div>

<style>
  .ticker-input {
    display: flex;
    align-items: center;
    gap: 0.8rem;
    background: var(--bg-card);
    border: 1px solid var(--border);
    border-radius: 6px;
    padding: 0.6rem 1rem;
    transition: border-color 0.2s;
  }

  .ticker-input:focus-within {
    border-color: var(--accent);
  }

  .prompt {
    color: var(--success);
    font-size: 1.1rem;
    font-weight: 600;
  }

  input {
    flex: 1;
    background: none;
    border: none;
    color: var(--text);
    font-size: 0.95rem;
    text-transform: uppercase;
    letter-spacing: 0.05em;
  }

  input::placeholder {
    color: var(--text-muted);
    text-transform: none;
    letter-spacing: normal;
  }

  input:disabled {
    opacity: 0.5;
    cursor: not-allowed;
  }

  button {
    background: var(--accent);
    color: var(--bg-dark);
    border: none;
    padding: 0.4rem 1rem;
    border-radius: 4px;
    font-size: 0.85rem;
    font-weight: 600;
    cursor: pointer;
    transition: all 0.2s;
    text-transform: uppercase;
    letter-spacing: 0.05em;
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
    .ticker-input {
      flex-wrap: wrap;
      padding: 0.8rem 1rem;
    }

    input {
      font-size: 16px; /* Prevents iOS zoom on focus */
      min-width: 0;
    }

    button {
      padding: 0.5rem 1.2rem;
    }
  }

  /* Small mobile breakpoint */
  @media (max-width: 480px) {
    .ticker-input {
      gap: 0.6rem;
      padding: 0.6rem 0.8rem;
    }

    .prompt {
      font-size: 1rem;
    }

    input {
      font-size: 16px;
    }

    button {
      padding: 0.45rem 0.9rem;
      font-size: 0.8rem;
    }
  }
</style>
