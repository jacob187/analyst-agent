<script lang="ts">
  import { createEventDispatcher } from 'svelte';

  const dispatch = createEventDispatcher<{ submit: { googleApiKey: string; secHeader: string; tavilyApiKey: string } }>();

  export let googleApiKey = '';
  export let secHeader = '';
  export let tavilyApiKey = '';
  let showKeys = false;

  function handleSubmit() {
    if (googleApiKey.trim() && secHeader.trim()) {
      dispatch('submit', { googleApiKey, secHeader, tavilyApiKey });
    }
  }

  function handleKeydown(e: KeyboardEvent) {
    if (e.key === 'Enter' && !e.shiftKey) {
      e.preventDefault();
      handleSubmit();
    }
  }
</script>

<div class="api-key-input">
  <div class="header">
    <h2>Configure API Access</h2>
    <p>Keys are stored locally in your database for convenience. Only enter once.</p>
  </div>

  <div class="input-group">
    <label for="google-api-key">
      <span class="label-text">Google API Key</span>
      <span class="label-hint">From Google AI Studio</span>
    </label>
    <div class="input-wrapper">
      <input
        id="google-api-key"
        type={showKeys ? 'text' : 'password'}
        bind:value={googleApiKey}
        on:keydown={handleKeydown}
        placeholder="AIza..."
        autocomplete="off"
      />
    </div>
  </div>

  <div class="input-group">
    <label for="sec-header">
      <span class="label-text">SEC User Agent</span>
      <span class="label-hint">Your name and email (required by SEC)</span>
    </label>
    <div class="input-wrapper">
      <input
        id="sec-header"
        type="text"
        bind:value={secHeader}
        on:keydown={handleKeydown}
        placeholder="Name your.email@example.com"
        autocomplete="email"
      />
    </div>
  </div>

  <div class="input-group optional">
    <label for="tavily-api-key">
      <span class="label-text">Tavily API Key <span class="optional-badge">Optional</span></span>
      <span class="label-hint">Enables web research, news, and competitor analysis</span>
    </label>
    <div class="input-wrapper">
      <input
        id="tavily-api-key"
        type={showKeys ? 'text' : 'password'}
        bind:value={tavilyApiKey}
        on:keydown={handleKeydown}
        placeholder="tvly-..."
        autocomplete="off"
      />
    </div>
  </div>

  <div class="actions">
    <label class="show-keys">
      <input type="checkbox" bind:checked={showKeys} />
      <span>Show keys</span>
    </label>

    <button
      on:click={handleSubmit}
      disabled={!googleApiKey.trim() || !secHeader.trim()}
      class="submit-btn"
    >
      Continue →
    </button>
  </div>

  <div class="help">
    <div class="help-item">
      <strong>Google API Key:</strong> Get one free at
      <a href="https://aistudio.google.com/app/apikey" target="_blank" rel="noopener">
        Google AI Studio
      </a>
    </div>
    <div class="help-item">
      <strong>SEC requires</strong> a valid email in the User-Agent header per their
      <a href="https://www.sec.gov/os/webmaster-faq#code-support" target="_blank" rel="noopener">
        fair access policy
      </a>
    </div>
    <div class="help-item">
      <strong>Tavily API Key (optional):</strong> Enables deep web research at
      <a href="https://app.tavily.com/sign-in" target="_blank" rel="noopener">
        Tavily
      </a>
    </div>
  </div>
</div>

<style>
  .api-key-input {
    max-width: 600px;
    background: var(--bg-card);
    border: 1px solid var(--border);
    border-radius: 8px;
    padding: 2rem;
  }

  .header {
    margin-bottom: 2rem;
  }

  .header h2 {
    font-size: 1.3rem;
    color: var(--text);
    margin-bottom: 0.5rem;
    font-weight: 600;
  }

  .header p {
    color: var(--text-dim);
    font-size: 0.85rem;
  }

  .input-group {
    margin-bottom: 1.5rem;
  }

  label {
    display: flex;
    flex-direction: column;
    gap: 0.3rem;
    margin-bottom: 0.5rem;
  }

  .label-text {
    color: var(--text);
    font-size: 0.9rem;
    font-weight: 500;
  }

  .label-hint {
    color: var(--text-muted);
    font-size: 0.75rem;
  }

  .optional-badge {
    background: rgba(0, 255, 157, 0.15);
    color: var(--accent);
    padding: 0.1rem 0.4rem;
    border-radius: 3px;
    font-size: 0.65rem;
    font-weight: 600;
    text-transform: uppercase;
    letter-spacing: 0.05em;
    margin-left: 0.4rem;
  }

  .input-group.optional {
    opacity: 0.9;
    border-left: 2px solid var(--border);
    padding-left: 1rem;
  }

  .input-wrapper {
    position: relative;
  }

  input[type="text"],
  input[type="password"] {
    width: 100%;
    background: var(--bg-input);
    border: 1px solid var(--border);
    border-radius: 4px;
    padding: 0.8rem;
    color: var(--text);
    font-family: inherit;
    font-size: 0.9rem;
    transition: border-color 0.2s;
  }

  input:focus {
    outline: none;
    border-color: var(--accent);
  }

  input::placeholder {
    color: var(--text-muted);
  }

  .actions {
    display: flex;
    justify-content: space-between;
    align-items: center;
    margin-top: 2rem;
    padding-top: 1.5rem;
    border-top: 1px solid var(--border);
  }

  .show-keys {
    display: flex;
    align-items: center;
    gap: 0.5rem;
    cursor: pointer;
    color: var(--text-dim);
    font-size: 0.85rem;
    flex-direction: row;
  }

  .show-keys input[type="checkbox"] {
    width: auto;
    cursor: pointer;
  }

  .submit-btn {
    background: var(--accent);
    color: var(--bg-dark);
    border: none;
    padding: 0.8rem 2rem;
    border-radius: 4px;
    font-size: 0.9rem;
    font-weight: 600;
    cursor: pointer;
    transition: all 0.2s;
  }

  .submit-btn:hover:not(:disabled) {
    background: var(--accent-dim);
    transform: translateY(-1px);
  }

  .submit-btn:disabled {
    opacity: 0.4;
    cursor: not-allowed;
  }

  .help {
    margin-top: 1.5rem;
    padding-top: 1.5rem;
    border-top: 1px solid var(--border);
    display: flex;
    flex-direction: column;
    gap: 0.8rem;
  }

  .help-item {
    color: var(--text-muted);
    font-size: 0.8rem;
    line-height: 1.5;
  }

  .help-item strong {
    color: var(--text-dim);
  }

  .help-item a {
    color: var(--accent);
    text-decoration: none;
  }

  .help-item a:hover {
    text-decoration: underline;
  }

  /* Tablet breakpoint */
  @media (max-width: 768px) {
    .api-key-input {
      padding: 1.5rem;
      max-width: 100%;
    }

    .header {
      margin-bottom: 1.5rem;
    }

    .header h2 {
      font-size: 1.2rem;
    }

    .actions {
      flex-direction: column;
      gap: 1rem;
      align-items: stretch;
    }

    .show-keys {
      justify-content: center;
    }

    .submit-btn {
      width: 100%;
      padding: 1rem 2rem;
    }

    input[type="text"],
    input[type="password"] {
      font-size: 16px; /* Prevents iOS zoom on focus */
    }
  }

  /* Small mobile breakpoint */
  @media (max-width: 480px) {
    .api-key-input {
      padding: 1rem;
      border-radius: 6px;
    }

    .header {
      margin-bottom: 1.2rem;
    }

    .header h2 {
      font-size: 1.1rem;
    }

    .header p {
      font-size: 0.8rem;
    }

    .input-group {
      margin-bottom: 1.2rem;
    }

    .label-text {
      font-size: 0.85rem;
    }

    .label-hint {
      font-size: 0.7rem;
    }

    input[type="text"],
    input[type="password"] {
      padding: 0.7rem;
    }

    .actions {
      margin-top: 1.5rem;
      padding-top: 1.2rem;
    }

    .submit-btn {
      padding: 0.9rem 1.5rem;
      font-size: 0.85rem;
    }

    .help {
      margin-top: 1.2rem;
      padding-top: 1.2rem;
    }

    .help-item {
      font-size: 0.75rem;
    }
  }
</style>
