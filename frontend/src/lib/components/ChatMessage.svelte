<script lang="ts">
  export let role: 'user' | 'assistant';
  export let content: string;
</script>

<div class="message {role}">
  <div class="header">
    <span class="label">{role === 'user' ? 'YOU' : 'ANALYST'}</span>
  </div>
  <div class="content">{content}</div>
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
