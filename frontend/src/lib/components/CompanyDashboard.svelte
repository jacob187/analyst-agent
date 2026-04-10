<script lang="ts">
  /**
   * CompanyDashboard — tabbed company view with Overview, Filings, and Chart+Chat.
   *
   * Overview loads on mount (fast, no LLM). Filings loads on tab click
   * (lazy — avoids LLM cost unless user wants it). Chart+Chat reuses
   * existing components unchanged.
   *
   * Svelte 4 syntax to match existing components.
   */
  import { onMount, createEventDispatcher } from 'svelte';
  import StockChart from './StockChart.svelte';
  import ChatWindow from './ChatWindow.svelte';

  export let ticker: string;
  export let apiBase: string;
  export let googleApiKey: string | null = null;
  export let openaiApiKey: string | null = null;
  export let anthropicApiKey: string | null = null;
  export let secHeader: string | null = null;
  export let tavilyApiKey: string = '';
  export let modelId: string = '';
  export let sessionId: string | null = null;

  const dispatch = createEventDispatcher<{ back: void }>();

  type Tab = 'overview' | 'filings' | 'chart';
  let activeTab: Tab = 'overview';

  // --- Overview state ---
  let profile: any = null;
  let profileLoading = true;
  let profileError = '';

  // --- Filings state ---
  let filings: any = null;
  let filingsLoading = false;
  let filingsLoaded = false;
  let filingsError = '';

  // Build headers for API requests (sends API keys from localStorage)
  function apiHeaders(): Record<string, string> {
    const h: Record<string, string> = {};
    if (googleApiKey) h['X-Google-Api-Key'] = googleApiKey;
    if (openaiApiKey) h['X-Openai-Api-Key'] = openaiApiKey;
    if (anthropicApiKey) h['X-Anthropic-Api-Key'] = anthropicApiKey;
    if (secHeader) h['X-Sec-Header'] = secHeader;
    if (tavilyApiKey) h['X-Tavily-Api-Key'] = tavilyApiKey;
    if (modelId) h['X-Model-Id'] = modelId;
    return h;
  }

  onMount(() => {
    fetchProfile();
  });

  async function fetchProfile() {
    profileLoading = true;
    profileError = '';
    try {
      const res = await fetch(`${apiBase}/api/company/${ticker}/profile`);
      if (!res.ok) throw new Error(`${res.status}: ${res.statusText}`);
      profile = await res.json();
    } catch (e: any) {
      profileError = e.message || 'Failed to load profile';
    } finally {
      profileLoading = false;
    }
  }

  async function fetchFilings() {
    if (filingsLoaded) return;
    filingsLoading = true;
    filingsError = '';
    try {
      const res = await fetch(`${apiBase}/api/company/${ticker}/filings`, {
        headers: apiHeaders(),
      });
      if (!res.ok) {
        const data = await res.json().catch(() => ({}));
        throw new Error(data.detail || `${res.status}: ${res.statusText}`);
      }
      filings = await res.json();
      filingsLoaded = true;
    } catch (e: any) {
      filingsError = e.message || 'Failed to load filings';
    } finally {
      filingsLoading = false;
    }
  }

  function selectTab(tab: Tab) {
    activeTab = tab;
    if (tab === 'filings' && !filingsLoaded && !filingsLoading) {
      fetchFilings();
    }
  }

  // --- Helpers ---
  function fmt(n: number | null | undefined, decimals = 2): string {
    if (n == null) return '--';
    return n.toLocaleString(undefined, { minimumFractionDigits: decimals, maximumFractionDigits: decimals });
  }

  function fmtLarge(n: number | null | undefined): string {
    if (n == null) return '--';
    if (n >= 1e12) return `$${(n / 1e12).toFixed(2)}T`;
    if (n >= 1e9) return `$${(n / 1e9).toFixed(2)}B`;
    if (n >= 1e6) return `$${(n / 1e6).toFixed(1)}M`;
    return `$${n.toLocaleString()}`;
  }

  function fmtPct(n: number | null | undefined): string {
    if (n == null) return '--';
    return `${n >= 0 ? '+' : ''}${n.toFixed(2)}%`;
  }

  function signalClass(signal: string | undefined): string {
    if (!signal) return '';
    const s = signal.toLowerCase();
    if (s === 'bullish' || s === 'oversold') return 'signal-bullish';
    if (s === 'bearish' || s === 'overbought') return 'signal-bearish';
    return 'signal-neutral';
  }
</script>

<div class="dashboard">
  <!-- Header -->
  <div class="dash-header">
    <div class="header-left">
      <button class="back-btn" on:click={() => dispatch('back')}>←</button>
      <div class="ticker-info">
        <h1 class="ticker-symbol">{ticker}</h1>
        {#if profile?.company?.name}
          <span class="company-name">{profile.company.name}</span>
        {/if}
      </div>
    </div>
    {#if profile?.quote?.price}
      <div class="price-block">
        <span class="price">${fmt(profile.quote.price)}</span>
        {#if profile.quote.changePercent != null}
          <span class="change" class:positive={profile.quote.changePercent >= 0} class:negative={profile.quote.changePercent < 0}>
            {fmtPct(profile.quote.changePercent)}
          </span>
        {/if}
      </div>
    {/if}
  </div>

  <!-- Subheader -->
  {#if profile?.company?.sector || profile?.earnings?.earnings_date}
    <div class="dash-subheader">
      {#if profile.company.sector}
        <span class="tag">{profile.company.sector}</span>
      {/if}
      {#if profile.company.industry}
        <span class="tag">{profile.company.industry}</span>
      {/if}
      {#if profile.earnings?.earnings_date}
        <span class="tag earnings-tag">Earnings: {profile.earnings.earnings_date}</span>
      {/if}
    </div>
  {/if}

  <!-- Tabs -->
  <div class="tabs">
    <button class="tab" class:active={activeTab === 'overview'} on:click={() => selectTab('overview')}>
      Overview
    </button>
    <button class="tab" class:active={activeTab === 'filings'} on:click={() => selectTab('filings')}>
      Filings
    </button>
    <button class="tab" class:active={activeTab === 'chart'} on:click={() => selectTab('chart')}>
      Chart + Chat
    </button>
  </div>

  <!-- Tab content -->
  <div class="tab-content">
    {#if activeTab === 'overview'}
      <!-- ── OVERVIEW TAB ─────────────────────────────────────────── -->
      {#if profileLoading}
        <div class="loading-state">Loading company data...</div>
      {:else if profileError}
        <div class="error-state">{profileError}</div>
      {:else if profile}
        <div class="overview-grid">
          <!-- Key Metrics -->
          <div class="card">
            <h3 class="card-title">Key Metrics</h3>
            <div class="metrics-grid">
              <div class="metric">
                <span class="metric-label">Market Cap</span>
                <span class="metric-value">{fmtLarge(profile.metrics?.market_cap)}</span>
              </div>
              <div class="metric">
                <span class="metric-label">P/E Ratio</span>
                <span class="metric-value">{fmt(profile.metrics?.pe_ratio, 1)}</span>
              </div>
              <div class="metric">
                <span class="metric-label">Forward P/E</span>
                <span class="metric-value">{fmt(profile.metrics?.forward_pe, 1)}</span>
              </div>
              <div class="metric">
                <span class="metric-label">Beta</span>
                <span class="metric-value">{fmt(profile.metrics?.beta)}</span>
              </div>
              <div class="metric">
                <span class="metric-label">52W High</span>
                <span class="metric-value">${fmt(profile.metrics?.['52wk_high'])}</span>
              </div>
              <div class="metric">
                <span class="metric-label">52W Low</span>
                <span class="metric-value">${fmt(profile.metrics?.['52wk_low'])}</span>
              </div>
              <div class="metric">
                <span class="metric-label">Div Yield</span>
                <span class="metric-value">{profile.metrics?.dividend_yield ? fmtPct(profile.metrics.dividend_yield * 100) : '--'}</span>
              </div>
              <div class="metric">
                <span class="metric-label">P/B Ratio</span>
                <span class="metric-value">{fmt(profile.metrics?.price_to_book, 1)}</span>
              </div>
            </div>
          </div>

          <!-- Technical Snapshot -->
          <div class="card">
            <h3 class="card-title">Technical Snapshot</h3>
            <div class="tech-grid">
              {#if profile.technicals?.rsi}
                <div class="tech-item">
                  <span class="tech-label">RSI</span>
                  <span class="tech-value">{fmt(profile.technicals.rsi.current, 1)}</span>
                  <span class="tech-signal {signalClass(profile.technicals.rsi.signal)}">{profile.technicals.rsi.signal}</span>
                </div>
              {/if}
              {#if profile.technicals?.macd}
                <div class="tech-item">
                  <span class="tech-label">MACD</span>
                  <span class="tech-value">{fmt(profile.technicals.macd.histogram, 3)}</span>
                  <span class="tech-signal {signalClass(profile.technicals.macd.signal)}">{profile.technicals.macd.signal}</span>
                </div>
              {/if}
              {#if profile.technicals?.adx}
                <div class="tech-item">
                  <span class="tech-label">ADX</span>
                  <span class="tech-value">{fmt(profile.technicals.adx.adx, 1)}</span>
                  <span class="tech-signal">{profile.technicals.adx.trend_strength || ''}</span>
                </div>
              {/if}
              {#if profile.technicals?.bollinger_bands}
                <div class="tech-item">
                  <span class="tech-label">Bollinger</span>
                  <span class="tech-value">{profile.technicals.bollinger_bands.position?.replace(/_/g, ' ') || '--'}</span>
                </div>
              {/if}
            </div>
          </div>

          <!-- Patterns -->
          {#if profile.patterns && profile.patterns.length > 0}
            <div class="card">
              <h3 class="card-title">Detected Patterns</h3>
              <div class="patterns-list">
                {#each profile.patterns as p}
                  <div class="pattern-item">
                    <span class="pattern-name">{p.type.replace(/_/g, ' ')}</span>
                    <span class="pattern-dir {signalClass(p.direction)}">{p.direction}</span>
                    <span class="pattern-conf">{Math.round(p.confidence * 100)}%</span>
                  </div>
                {/each}
              </div>
            </div>
          {/if}

          <!-- Market Regime -->
          {#if profile.regime?.trend}
            <div class="card">
              <h3 class="card-title">Market Regime</h3>
              <div class="regime-badges">
                <span class="badge badge-{profile.regime.trend}">{profile.regime.trend}</span>
                <span class="badge">{profile.regime.volatility} vol</span>
                <span class="badge">{profile.regime.phase}</span>
              </div>
            </div>
          {/if}

          <!-- Company Summary -->
          {#if profile.company?.summary}
            <div class="card card-full">
              <h3 class="card-title">About</h3>
              <p class="summary-text">{profile.company.summary}</p>
              <div class="company-meta">
                {#if profile.company.employees}
                  <span>{profile.company.employees.toLocaleString()} employees</span>
                {/if}
                {#if profile.company.country}
                  <span>{profile.company.country}</span>
                {/if}
                {#if profile.company.website}
                  <a href={profile.company.website} target="_blank" rel="noopener">{profile.company.website}</a>
                {/if}
              </div>
            </div>
          {/if}
        </div>
      {/if}

    {:else if activeTab === 'filings'}
      <!-- ── FILINGS TAB ──────────────────────────────────────────── -->
      {#if filingsLoading}
        <div class="loading-state">
          <div class="spinner"></div>
          <p>Analyzing SEC filings... This may take a moment on first load.</p>
          <p class="loading-hint">Results are cached — future loads will be instant.</p>
        </div>
      {:else if filingsError}
        <div class="error-state">
          <p>{filingsError}</p>
          <button class="btn" on:click={() => { filingsLoaded = false; fetchFilings(); }}>Retry</button>
        </div>
      {:else if filings}
        <div class="filings-content">
          <!-- 10-K or 20-F annual report -->
          {#if filings.tenk}
            <div class="filing-section">
              <div class="filing-header">
                <h3>{filings.tenk.metadata?.form || '10-K'} Annual Report</h3>
                <div class="filing-meta">
                  <span>Filed: {filings.tenk.metadata?.filing_date}</span>
                  <span>Period: {filings.tenk.metadata?.period_of_report}</span>
                  {#if filings.tenk.metadata?.edgar_url}
                    <a href={filings.tenk.metadata.edgar_url} target="_blank" rel="noopener" class="edgar-link">View on EDGAR</a>
                  {/if}
                </div>
              </div>

              {#if filings.tenk.risk_10k}
                <div class="analysis-card">
                  <h4>Risk Factors</h4>
                  <p>{filings.tenk.risk_10k.summary}</p>
                  {#if filings.tenk.risk_10k.key_risks?.length}
                    <ul class="analysis-list">
                      {#each filings.tenk.risk_10k.key_risks as risk}
                        <li>{risk}</li>
                      {/each}
                    </ul>
                  {/if}
                  <div class="sentiment-badge" class:positive={filings.tenk.risk_10k.sentiment_score > 0} class:negative={filings.tenk.risk_10k.sentiment_score <= 0}>
                    Sentiment: {fmt(filings.tenk.risk_10k.sentiment_score, 1)}
                  </div>
                </div>
              {/if}

              {#if filings.tenk.mda_10k}
                <div class="analysis-card">
                  <h4>Management Discussion & Analysis</h4>
                  <p>{filings.tenk.mda_10k.summary}</p>
                  {#if filings.tenk.mda_10k.future_outlook}
                    <div class="outlook">
                      <strong>Outlook:</strong> {filings.tenk.mda_10k.future_outlook}
                    </div>
                  {/if}
                  <div class="sentiment-badge" class:positive={filings.tenk.mda_10k.sentiment_score > 0} class:negative={filings.tenk.mda_10k.sentiment_score <= 0}>
                    Sentiment: {fmt(filings.tenk.mda_10k.sentiment_score, 1)}
                  </div>
                </div>
              {/if}

              {#if filings.tenk.balance}
                <div class="analysis-card">
                  <h4>Balance Sheet</h4>
                  <p>{filings.tenk.balance.summary}</p>
                  {#if filings.tenk.balance.red_flags?.length}
                    <div class="red-flags">
                      <strong>Red Flags:</strong>
                      <ul>
                        {#each filings.tenk.balance.red_flags as flag}
                          <li>{flag}</li>
                        {/each}
                      </ul>
                    </div>
                  {/if}
                </div>
              {/if}
            </div>
          {/if}

          <!-- 10-Q -->
          {#if filings.tenq}
            <div class="filing-section">
              <div class="filing-header">
                <h3>10-Q Quarterly Report</h3>
                <div class="filing-meta">
                  <span>Filed: {filings.tenq.metadata?.filing_date}</span>
                  <span>Period: {filings.tenq.metadata?.period_of_report}</span>
                  {#if filings.tenq.metadata?.edgar_url}
                    <a href={filings.tenq.metadata.edgar_url} target="_blank" rel="noopener" class="edgar-link">View on EDGAR</a>
                  {/if}
                </div>
              </div>

              {#if filings.tenq.risk_10q}
                <div class="analysis-card">
                  <h4>Risk Factors (Quarterly)</h4>
                  <p>{filings.tenq.risk_10q.summary}</p>
                </div>
              {/if}

              {#if filings.tenq.mda_10q}
                <div class="analysis-card">
                  <h4>MD&A (Quarterly)</h4>
                  <p>{filings.tenq.mda_10q.summary}</p>
                  {#if filings.tenq.mda_10q.future_outlook}
                    <div class="outlook">
                      <strong>Outlook:</strong> {filings.tenq.mda_10q.future_outlook}
                    </div>
                  {/if}
                </div>
              {/if}
            </div>
          {/if}

          <!-- 8-K Earnings -->
          {#if filings.earnings?.has_earnings}
            <div class="filing-section earnings-section">
              <div class="filing-header">
                <h3>8-K Earnings Release</h3>
                <div class="filing-meta">
                  <span>Filed: {filings.earnings.metadata?.filing_date}</span>
                  {#if filings.earnings.metadata?.edgar_url}
                    <a href={filings.earnings.metadata.edgar_url} target="_blank" rel="noopener" class="edgar-link">View on EDGAR</a>
                  {/if}
                </div>
              </div>

              {#if filings.earnings.analysis}
                <div class="analysis-card">
                  <p class="earnings-summary">{filings.earnings.analysis.summary}</p>
                  {#if filings.earnings.analysis.key_metrics?.length}
                    <div class="metrics-pills">
                      {#each filings.earnings.analysis.key_metrics as metric}
                        <span class="pill">{metric}</span>
                      {/each}
                    </div>
                  {/if}
                  {#if filings.earnings.analysis.beats_misses?.length}
                    <div class="beats-misses">
                      {#each filings.earnings.analysis.beats_misses as bm}
                        <span class="beat-miss">{bm}</span>
                      {/each}
                    </div>
                  {/if}
                  {#if filings.earnings.analysis.guidance}
                    <div class="outlook">
                      <strong>Guidance:</strong> {filings.earnings.analysis.guidance}
                    </div>
                  {/if}
                  <div class="sentiment-badge" class:positive={filings.earnings.analysis.sentiment_score > 0} class:negative={filings.earnings.analysis.sentiment_score <= 0}>
                    Sentiment: {fmt(filings.earnings.analysis.sentiment_score, 1)}
                  </div>
                </div>
              {/if}
            </div>
          {:else if filings.earnings}
            <div class="filing-section">
              <div class="filing-header">
                <h3>8-K Earnings</h3>
              </div>
              <p class="no-data">No recent earnings release found.</p>
            </div>
          {/if}

          {#if !filings.tenk && !filings.tenq}
            <div class="no-data">No SEC filings available for {ticker}.</div>
          {/if}
        </div>
      {:else}
        <div class="empty-state">
          <p>Click the Filings tab to analyze SEC filings for {ticker}.</p>
        </div>
      {/if}

    {:else if activeTab === 'chart'}
      <!-- ── CHART + CHAT TAB ─────────────────────────────────────── -->
      <div class="chart-chat-layout">
        <div class="chart-col">
          <StockChart {ticker} />
        </div>
        <div class="chat-col">
          <ChatWindow
            {ticker}
            {googleApiKey}
            {openaiApiKey}
            {anthropicApiKey}
            {secHeader}
            {tavilyApiKey}
            {modelId}
            {sessionId}
          />
        </div>
      </div>
    {/if}
  </div>
</div>

<style>
  .dashboard {
    display: flex;
    flex-direction: column;
    height: 100%;
    min-height: 0;
  }

  /* ── Header ───────────────────────────────────────────── */
  .dash-header {
    display: flex;
    align-items: center;
    justify-content: space-between;
    padding: 0 0 0.5rem;
    flex-shrink: 0;
  }

  .header-left {
    display: flex;
    align-items: center;
    gap: 0.8rem;
  }

  .back-btn {
    background: none;
    border: 1px solid var(--border);
    color: var(--text-dim);
    font-size: 1.1rem;
    cursor: pointer;
    padding: 0.3rem 0.6rem;
    border-radius: 4px;
    transition: all 0.2s;
  }

  .back-btn:hover {
    color: var(--accent);
    border-color: var(--accent);
  }

  .ticker-info {
    display: flex;
    align-items: baseline;
    gap: 0.6rem;
  }

  .ticker-symbol {
    font-size: 1.5rem;
    font-weight: 700;
    color: var(--text);
    margin: 0;
    letter-spacing: 0.05em;
  }

  .company-name {
    color: var(--text-dim);
    font-size: 0.9rem;
  }

  .price-block {
    display: flex;
    align-items: baseline;
    gap: 0.6rem;
  }

  .price {
    font-size: 1.4rem;
    font-weight: 700;
    color: var(--text);
  }

  .change {
    font-size: 0.95rem;
    font-weight: 600;
  }

  .change.positive { color: #00e676; }
  .change.negative { color: #ff5252; }

  /* ── Subheader ────────────────────────────────────────── */
  .dash-subheader {
    display: flex;
    gap: 0.5rem;
    flex-wrap: wrap;
    padding-bottom: 0.8rem;
    flex-shrink: 0;
  }

  .tag {
    padding: 0.2rem 0.6rem;
    background: var(--bg-card);
    border: 1px solid var(--border);
    border-radius: 3px;
    color: var(--text-dim);
    font-size: 0.75rem;
    letter-spacing: 0.03em;
  }

  .earnings-tag {
    color: var(--accent);
    border-color: var(--accent);
    background: rgba(0, 212, 255, 0.08);
  }

  /* ── Tabs ──────────────────────────────────────────────── */
  .tabs {
    display: flex;
    gap: 0;
    border-bottom: 1px solid var(--border);
    margin-bottom: 0.8rem;
    flex-shrink: 0;
  }

  .tab {
    background: none;
    border: none;
    border-bottom: 2px solid transparent;
    color: var(--text-dim);
    font-family: inherit;
    font-size: 0.85rem;
    padding: 0.6rem 1.2rem;
    cursor: pointer;
    transition: color 0.2s, border-color 0.2s;
    letter-spacing: 0.04em;
  }

  .tab:hover { color: var(--text); }

  .tab.active {
    color: var(--accent);
    border-bottom-color: var(--accent);
  }

  /* ── Tab content ──────────────────────────────────────── */
  .tab-content {
    flex: 1;
    min-height: 0;
    overflow-y: auto;
  }

  /* ── Overview grid ────────────────────────────────────── */
  .overview-grid {
    display: grid;
    grid-template-columns: 1fr 1fr;
    gap: 0.8rem;
    padding-bottom: 1rem;
  }

  .card {
    background: var(--bg-card);
    border: 1px solid var(--border);
    border-radius: 6px;
    padding: 1rem;
  }

  .card-full {
    grid-column: 1 / -1;
  }

  .card-title {
    color: var(--text);
    font-size: 0.8rem;
    font-weight: 600;
    text-transform: uppercase;
    letter-spacing: 0.08em;
    margin: 0 0 0.8rem;
    padding-bottom: 0.4rem;
    border-bottom: 1px solid var(--border);
  }

  /* Metrics grid */
  .metrics-grid {
    display: grid;
    grid-template-columns: 1fr 1fr;
    gap: 0.6rem;
  }

  .metric {
    display: flex;
    flex-direction: column;
    gap: 0.15rem;
  }

  .metric-label {
    color: var(--text-muted);
    font-size: 0.7rem;
    text-transform: uppercase;
    letter-spacing: 0.06em;
  }

  .metric-value {
    color: var(--text);
    font-size: 0.95rem;
    font-weight: 600;
  }

  /* Technical items */
  .tech-grid {
    display: flex;
    flex-direction: column;
    gap: 0.6rem;
  }

  .tech-item {
    display: flex;
    align-items: center;
    gap: 0.6rem;
  }

  .tech-label {
    color: var(--text-muted);
    font-size: 0.75rem;
    text-transform: uppercase;
    letter-spacing: 0.05em;
    min-width: 5rem;
  }

  .tech-value {
    color: var(--text);
    font-size: 0.9rem;
    font-weight: 600;
    min-width: 4rem;
  }

  .tech-signal {
    font-size: 0.75rem;
    font-weight: 600;
    text-transform: uppercase;
    letter-spacing: 0.05em;
  }

  .signal-bullish { color: #00e676; }
  .signal-bearish { color: #ff5252; }
  .signal-neutral { color: var(--text-dim); }

  /* Patterns */
  .patterns-list {
    display: flex;
    flex-direction: column;
    gap: 0.5rem;
  }

  .pattern-item {
    display: flex;
    align-items: center;
    gap: 0.8rem;
    padding: 0.4rem 0;
    border-bottom: 1px solid var(--border);
  }

  .pattern-item:last-child { border-bottom: none; }

  .pattern-name {
    color: var(--text);
    font-size: 0.85rem;
    text-transform: capitalize;
    flex: 1;
  }

  .pattern-dir {
    font-size: 0.75rem;
    font-weight: 600;
    text-transform: uppercase;
  }

  .pattern-conf {
    color: var(--text-dim);
    font-size: 0.8rem;
  }

  /* Regime badges */
  .regime-badges {
    display: flex;
    gap: 0.5rem;
    flex-wrap: wrap;
  }

  .badge {
    padding: 0.3rem 0.7rem;
    border-radius: 3px;
    font-size: 0.75rem;
    font-weight: 600;
    text-transform: uppercase;
    letter-spacing: 0.05em;
    background: var(--bg-darker);
    color: var(--text-dim);
    border: 1px solid var(--border);
  }

  .badge-bull { color: #00e676; border-color: #00e676; background: rgba(0, 230, 118, 0.1); }
  .badge-bear { color: #ff5252; border-color: #ff5252; background: rgba(255, 82, 82, 0.1); }

  /* Summary */
  .summary-text {
    color: var(--text-dim);
    font-size: 0.85rem;
    line-height: 1.6;
    margin: 0 0 0.8rem;
  }

  .company-meta {
    display: flex;
    gap: 1rem;
    flex-wrap: wrap;
    font-size: 0.75rem;
    color: var(--text-muted);
  }

  .company-meta a {
    color: var(--accent);
    text-decoration: none;
  }

  .company-meta a:hover { text-decoration: underline; }

  /* ── Filings ──────────────────────────────────────────── */
  .filings-content {
    display: flex;
    flex-direction: column;
    gap: 1.2rem;
    padding-bottom: 1rem;
  }

  .filing-section {
    background: var(--bg-card);
    border: 1px solid var(--border);
    border-radius: 6px;
    padding: 1rem;
  }

  .earnings-section {
    border-color: var(--accent);
    background: rgba(0, 212, 255, 0.03);
  }

  .filing-header {
    margin-bottom: 0.8rem;
  }

  .filing-header h3 {
    color: var(--text);
    font-size: 1rem;
    font-weight: 600;
    margin: 0 0 0.4rem;
  }

  .filing-meta {
    display: flex;
    gap: 1rem;
    flex-wrap: wrap;
    font-size: 0.75rem;
    color: var(--text-muted);
  }

  .edgar-link {
    color: var(--accent);
    text-decoration: none;
    font-weight: 600;
  }

  .edgar-link:hover { text-decoration: underline; }

  .analysis-card {
    padding: 0.8rem 0;
    border-top: 1px solid var(--border);
  }

  .analysis-card h4 {
    color: var(--text);
    font-size: 0.85rem;
    font-weight: 600;
    margin: 0 0 0.5rem;
  }

  .analysis-card p {
    color: var(--text-dim);
    font-size: 0.85rem;
    line-height: 1.6;
    margin: 0 0 0.5rem;
  }

  .analysis-list {
    margin: 0.5rem 0;
    padding-left: 1.2rem;
    color: var(--text-dim);
    font-size: 0.82rem;
    line-height: 1.5;
  }

  .sentiment-badge {
    display: inline-block;
    padding: 0.2rem 0.6rem;
    border-radius: 3px;
    font-size: 0.72rem;
    font-weight: 600;
    margin-top: 0.4rem;
    background: var(--bg-darker);
    color: var(--text-dim);
    border: 1px solid var(--border);
  }

  .sentiment-badge.positive { color: #00e676; border-color: #00e676; }
  .sentiment-badge.negative { color: #ff5252; border-color: #ff5252; }

  .outlook {
    color: var(--text-dim);
    font-size: 0.82rem;
    line-height: 1.5;
    margin: 0.5rem 0;
    padding: 0.5rem;
    background: var(--bg-darker);
    border-radius: 4px;
    border-left: 2px solid var(--accent);
  }

  .earnings-summary {
    font-size: 0.9rem !important;
    color: var(--text) !important;
  }

  .metrics-pills {
    display: flex;
    gap: 0.4rem;
    flex-wrap: wrap;
    margin: 0.5rem 0;
  }

  .pill {
    padding: 0.2rem 0.5rem;
    background: var(--bg-darker);
    border: 1px solid var(--border);
    border-radius: 3px;
    font-size: 0.75rem;
    color: var(--text-dim);
  }

  .beats-misses {
    display: flex;
    flex-direction: column;
    gap: 0.3rem;
    margin: 0.5rem 0;
  }

  .beat-miss {
    color: var(--text-dim);
    font-size: 0.82rem;
    padding-left: 0.8rem;
    border-left: 2px solid #00e676;
  }

  .red-flags {
    margin-top: 0.5rem;
    padding: 0.5rem;
    background: rgba(255, 82, 82, 0.05);
    border: 1px solid rgba(255, 82, 82, 0.2);
    border-radius: 4px;
    font-size: 0.82rem;
    color: var(--text-dim);
  }

  .red-flags strong { color: #ff5252; }
  .red-flags ul { margin: 0.3rem 0 0; padding-left: 1.2rem; }

  .no-data {
    color: var(--text-muted);
    font-size: 0.85rem;
    padding: 1rem 0;
  }

  /* ── Chart + Chat layout ──────────────────────────────── */
  .chart-chat-layout {
    display: grid;
    grid-template-columns: 3fr 2fr;
    gap: 1rem;
    height: 100%;
    min-height: 0;
  }

  .chart-col {
    overflow-y: auto;
    min-height: 0;
  }

  .chat-col {
    overflow: hidden;
    min-height: 0;
    display: flex;
    flex-direction: column;
  }

  /* ── Loading / error states ───────────────────────────── */
  .loading-state, .error-state, .empty-state {
    display: flex;
    flex-direction: column;
    align-items: center;
    justify-content: center;
    padding: 3rem 1rem;
    color: var(--text-dim);
    text-align: center;
    gap: 0.8rem;
  }

  .loading-hint {
    color: var(--text-muted);
    font-size: 0.8rem;
  }

  .spinner {
    width: 24px;
    height: 24px;
    border: 2px solid var(--border);
    border-top-color: var(--accent);
    border-radius: 50%;
    animation: spin 0.8s linear infinite;
  }

  @keyframes spin {
    to { transform: rotate(360deg); }
  }

  .btn {
    background: var(--bg-card);
    color: var(--text-dim);
    border: 1px solid var(--border);
    padding: 0.5rem 1rem;
    border-radius: 4px;
    font-family: inherit;
    font-size: 0.85rem;
    cursor: pointer;
    transition: all 0.2s;
  }

  .btn:hover {
    color: var(--text);
    border-color: var(--accent);
  }

  /* ── Responsive ───────────────────────────────────────── */
  @media (max-width: 768px) {
    .overview-grid {
      grid-template-columns: 1fr;
    }

    .chart-chat-layout {
      grid-template-columns: 1fr;
      height: auto;
    }

    .chart-col {
      max-height: 45vw;
      min-height: 220px;
    }

    .chat-col {
      height: 55vh;
      min-height: 320px;
    }

    .dash-header {
      flex-direction: column;
      align-items: flex-start;
      gap: 0.5rem;
    }

    .ticker-symbol { font-size: 1.2rem; }
    .price { font-size: 1.1rem; }
  }
</style>
