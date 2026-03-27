<script lang="ts">
  /**
   * ChartControls — toggle indicator visibility and switch timeframes.
   * Uses Svelte 5 runes syntax.
   */

  interface Props {
    period: string;
    onPeriodChange: (period: string) => void;
    showMAs: boolean;
    onToggleMAs: () => void;
    showBollinger: boolean;
    onToggleBollinger: () => void;
    showRSI: boolean;
    onToggleRSI: () => void;
    showMACD: boolean;
    onToggleMACD: () => void;
  }

  let {
    period,
    onPeriodChange,
    showMAs,
    onToggleMAs,
    showBollinger,
    onToggleBollinger,
    showRSI,
    onToggleRSI,
    showMACD,
    onToggleMACD,
  }: Props = $props();

  const timeframes = [
    { label: "1W", value: "1w" },
    { label: "1M", value: "1mo" },
    { label: "3M", value: "3mo" },
    { label: "6M", value: "6mo" },
    { label: "1Y", value: "1y" },
  ];
</script>

<div class="chart-controls">
  <div class="timeframes">
    {#each timeframes as tf}
      <button
        class="tf-btn"
        class:active={period === tf.value}
        onclick={() => onPeriodChange(tf.value)}
      >
        {tf.label}
      </button>
    {/each}
  </div>

  <div class="indicators">
    <button class="ind-btn" class:active={showMAs} onclick={onToggleMAs}>
      MA
    </button>
    <button class="ind-btn" class:active={showBollinger} onclick={onToggleBollinger}>
      BB
    </button>
    <button class="ind-btn" class:active={showRSI} onclick={onToggleRSI}>
      RSI
    </button>
    <button class="ind-btn" class:active={showMACD} onclick={onToggleMACD}>
      MACD
    </button>
  </div>
</div>

<style>
  .chart-controls {
    display: flex;
    justify-content: space-between;
    align-items: center;
    padding: 0.5rem 0.75rem;
    background: var(--bg-card);
    border: 1px solid var(--border);
    border-radius: 6px 6px 0 0;
    gap: 0.5rem;
    flex-wrap: wrap;
  }

  .timeframes,
  .indicators {
    display: flex;
    gap: 0.25rem;
  }

  .tf-btn,
  .ind-btn {
    background: transparent;
    border: 1px solid var(--border);
    color: var(--text-dim);
    padding: 0.25rem 0.6rem;
    border-radius: 4px;
    font-family: inherit;
    font-size: 0.75rem;
    cursor: pointer;
    transition: all 0.15s;
    letter-spacing: 0.03em;
  }

  .tf-btn:hover,
  .ind-btn:hover {
    border-color: var(--accent);
    color: var(--accent);
  }

  .tf-btn.active {
    background: var(--accent);
    color: var(--bg-dark);
    border-color: var(--accent);
    font-weight: 600;
  }

  .ind-btn.active {
    border-color: var(--accent);
    color: var(--accent);
  }

  @media (max-width: 600px) {
    .chart-controls {
      flex-direction: column;
      align-items: flex-start;
    }
  }
</style>
