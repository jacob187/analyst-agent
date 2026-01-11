<script lang="ts">
  import { createEventDispatcher } from 'svelte';

  const dispatch = createEventDispatcher<{ submit: { googleApiKey: string; secHeader: string } }>();

  let googleApiKey = '';
  let secHeader = '';
  let showKeys = false;

  function handleSubmit() {
    if (googleApiKey.trim() && secHeader.trim()) {
      dispatch('submit', { googleApiKey, secHeader });
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
    <p>Your keys are only sent to your self-hosted backend. Never stored.</p>
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
      <span class="label-hint">Your email address (required by SEC)</span>
    </label>
    <div class="input-wrapper">
      <input
        id="sec-header"
        type="text"
        bind:value={secHeader}
        on:keydown={handleKeydown}
        placeholder="your.email@example.com"
        autocomplete="email"
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
</style>
