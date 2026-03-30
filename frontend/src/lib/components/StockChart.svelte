<script lang="ts">
  /**
   * StockChart — interactive candlestick chart with technical indicators.
   *
   * Fetches OHLCV + indicator data from the backend REST endpoint and renders
   * using TradingView Lightweight Charts. All indicator math happens on the
   * backend — this component only renders.
   *
   * Uses Svelte 5 runes syntax ($state, $effect, $props).
   */
  import { onMount } from "svelte";
  import {
    createChart,
    CandlestickSeries,
    HistogramSeries,
    LineSeries,
    ColorType,
    LineStyle,
    type IChartApi,
    type ISeriesApi,
  } from "lightweight-charts";
  import ChartControls from "./ChartControls.svelte";

  // ── Props ────────────────────────────────────────────────────────────
  interface Props {
    ticker: string;
  }
  let { ticker }: Props = $props();

  // ── State ────────────────────────────────────────────────────────────
  let period = $state("1y");
  let loading = $state(true);
  let error = $state("");

  // Indicator toggles
  let showMAs = $state(true);
  let showBollinger = $state(true);
  let showRSI = $state(true);
  let showMACD = $state(true);

  // Track which indicators have data (short timeframes lack long MAs)
  let hasMA20 = $state(false);
  let hasMA50 = $state(false);
  let hasMA200 = $state(false);
  let hasBollinger = $state(false);

  // Current price info for the header
  let currentPrice = $state(0);
  let priceChange = $state(0);
  let priceChangePercent = $state(0);
  let priceDirection = $state<"up" | "down">("up");

  // ── DOM refs ─────────────────────────────────────────────────────────
  let priceContainer: HTMLDivElement;
  let rsiContainer: HTMLDivElement = $state(undefined as any);
  let macdContainer: HTMLDivElement = $state(undefined as any);

  // ── Chart instances ──────────────────────────────────────────────────
  let priceChart: IChartApi | null = null;
  let rsiChart: IChartApi | null = null;
  let macdChart: IChartApi | null = null;

  // Series references for updating data
  let candleSeries: ISeriesApi<"Candlestick"> | null = null;
  let volumeSeries: ISeriesApi<"Histogram"> | null = null;

  let ma20Series: ISeriesApi<"Line"> | null = null;
  let ma50Series: ISeriesApi<"Line"> | null = null;
  let ma200Series: ISeriesApi<"Line"> | null = null;

  let bbUpperSeries: ISeriesApi<"Line"> | null = null;
  let bbMiddleSeries: ISeriesApi<"Line"> | null = null;
  let bbLowerSeries: ISeriesApi<"Line"> | null = null;

  let rsiSeries: ISeriesApi<"Line"> | null = null;

  let macdLineSeries: ISeriesApi<"Line"> | null = null;
  let macdSignalSeries: ISeriesApi<"Line"> | null = null;
  let macdHistSeries: ISeriesApi<"Histogram"> | null = null;

  let resizeObserver: ResizeObserver | null = null;

  const API_BASE = import.meta.env.VITE_API_URL || "http://localhost:8000";

  // ── Chart theme ──────────────────────────────────────────────────────
  const CHART_BG = "#070a1a"; // --bg-darker
  const GRID_COLOR = "#1a1f3a";
  const TEXT_COLOR = "#a0a3b5"; // --text-dim
  const BORDER_COLOR = "#2a2f4a"; // --border

  // On narrow screens (mobile), disable vertical touch drag on charts so that
  // swiping up/down scrolls the page instead of zooming the chart. Horizontal
  // drag still pans the time axis. Pinch-to-zoom is also disabled on mobile to
  // avoid accidental zooms while scrolling.
  const isMobile = () => window.innerWidth <= 768;

  function chartOptions(height: number) {
    const mobile = isMobile();
    return {
      layout: {
        background: { type: ColorType.Solid as const, color: CHART_BG },
        textColor: TEXT_COLOR,
        fontSize: 11,
      },
      grid: {
        vertLines: { color: GRID_COLOR },
        horzLines: { color: GRID_COLOR },
      },
      rightPriceScale: { borderColor: BORDER_COLOR },
      timeScale: {
        borderColor: BORDER_COLOR,
        timeVisible: false,
      },
      crosshair: { mode: 0 },
      handleScroll: {
        vertTouchDrag: !mobile,
      },
      handleScale: {
        axisPressedMouseMove: !mobile,
        pinch: !mobile,
      },
      height,
    };
  }

  // ── Initialize charts ────────────────────────────────────────────────

  function initCharts() {
    if (!priceContainer) return;

    // Price chart (candlestick + volume + overlays)
    priceChart = createChart(priceContainer, {
      ...chartOptions(400),
      timeScale: {
        borderColor: BORDER_COLOR,
        timeVisible: false,
      },
    });

    candleSeries = priceChart.addSeries(CandlestickSeries, {
      upColor: "#26a69a",
      downColor: "#ef5350",
      borderVisible: false,
      wickUpColor: "#26a69a",
      wickDownColor: "#ef5350",
      lastValueVisible: false,
    });

    volumeSeries = priceChart.addSeries(HistogramSeries, {
      priceFormat: { type: "volume" },
      priceScaleId: "volume",
      lastValueVisible: false,
      priceLineVisible: false,
    });
    priceChart.priceScale("volume").applyOptions({
      scaleMargins: { top: 0.85, bottom: 0 },
    });

    // Shared options to keep the y-axis clean: hide per-series price labels
    // and price lines. Only the candlestick series shows its label.
    const cleanOverlay = { lastValueVisible: false, priceLineVisible: false };

    // MA overlays
    ma20Series = priceChart.addSeries(LineSeries, {
      color: "#2196F3",
      lineWidth: 1,
      visible: showMAs,
      ...cleanOverlay,
    });
    ma50Series = priceChart.addSeries(LineSeries, {
      color: "#FF9800",
      lineWidth: 1,
      visible: showMAs,
      ...cleanOverlay,
    });
    ma200Series = priceChart.addSeries(LineSeries, {
      color: "#F44336",
      lineWidth: 1,
      visible: showMAs,
      ...cleanOverlay,
    });

    // Bollinger Bands
    bbUpperSeries = priceChart.addSeries(LineSeries, {
      color: "rgba(0, 212, 255, 0.3)",
      lineWidth: 1,
      lineStyle: LineStyle.Dashed,
      visible: showBollinger,
      ...cleanOverlay,
    });
    bbMiddleSeries = priceChart.addSeries(LineSeries, {
      color: "rgba(0, 212, 255, 0.15)",
      lineWidth: 1,
      visible: showBollinger,
      ...cleanOverlay,
    });
    bbLowerSeries = priceChart.addSeries(LineSeries, {
      color: "rgba(0, 212, 255, 0.3)",
      lineWidth: 1,
      lineStyle: LineStyle.Dashed,
      visible: showBollinger,
      ...cleanOverlay,
    });

    // RSI sub-chart
    if (rsiContainer) {
      rsiChart = createChart(rsiContainer, chartOptions(120));

      rsiSeries = rsiChart.addSeries(LineSeries, {
        color: "#7E57C2",
        lineWidth: 1,
        title: "RSI",
      });

      // Overbought/oversold reference lines
      const rsiOverbought = rsiChart.addSeries(LineSeries, {
        color: "rgba(239, 83, 80, 0.4)",
        lineWidth: 1,
        lineStyle: LineStyle.Dotted,
        priceLineVisible: false,
        lastValueVisible: false,
        crosshairMarkerVisible: false,
      });
      const rsiOversold = rsiChart.addSeries(LineSeries, {
        color: "rgba(38, 166, 154, 0.4)",
        lineWidth: 1,
        lineStyle: LineStyle.Dotted,
        priceLineVisible: false,
        lastValueVisible: false,
        crosshairMarkerVisible: false,
      });

      // Store these for setting data later
      (rsiChart as any)._overbought = rsiOverbought;
      (rsiChart as any)._oversold = rsiOversold;
    }

    // MACD sub-chart
    if (macdContainer) {
      macdChart = createChart(macdContainer, chartOptions(120));

      // Histogram on an overlay scale so it doesn't interfere with line
      // rendering — same pattern as volume on the price chart. Both scales
      // auto-center around zero so they visually align.
      macdHistSeries = macdChart.addSeries(HistogramSeries, {
        priceScaleId: "macd_hist",
        color: "#26a69a",
        lastValueVisible: false,
        priceLineVisible: false,
      });
      macdChart.priceScale("macd_hist").applyOptions({
        scaleMargins: { top: 0.1, bottom: 0.1 },
      });

      // Line series on the default right scale — drawn on top of histogram
      macdLineSeries = macdChart.addSeries(LineSeries, {
        color: "#2962FF",
        lineWidth: 1,
        title: "MACD (12, 26, 9)",
        lastValueVisible: false,
        priceLineVisible: false,
      });
      macdSignalSeries = macdChart.addSeries(LineSeries, {
        color: "#FF6D00",
        lineWidth: 1,
        title: "Signal",
        lastValueVisible: false,
        priceLineVisible: false,
      });
    }

    // Sync time scales across charts
    syncCharts();

    // Resize observer
    resizeObserver = new ResizeObserver(() => {
      if (priceContainer && priceChart) {
        priceChart.applyOptions({ width: priceContainer.clientWidth });
      }
      if (rsiContainer && rsiChart) {
        rsiChart.applyOptions({ width: rsiContainer.clientWidth });
      }
      if (macdContainer && macdChart) {
        macdChart.applyOptions({ width: macdContainer.clientWidth });
      }
    });
    resizeObserver.observe(priceContainer);
  }

  // Guard flag prevents recursive sync: chart A scrolls -> updates B ->
  // B's handler fires -> tries to update A again. Without this flag each
  // scroll event would bounce between charts until the call stack settles.
  let syncing = false;

  function syncCharts() {
    if (!priceChart) return;

    priceChart
      .timeScale()
      .subscribeVisibleLogicalRangeChange((range) => {
        if (syncing || !range) return;
        syncing = true;
        rsiChart?.timeScale().setVisibleLogicalRange(range);
        macdChart?.timeScale().setVisibleLogicalRange(range);
        syncing = false;
      });

    // Sub-charts sync back to price chart (and each other)
    const subSync = (range: any) => {
      if (syncing || !range) return;
      syncing = true;
      priceChart?.timeScale().setVisibleLogicalRange(range);
      rsiChart?.timeScale().setVisibleLogicalRange(range);
      macdChart?.timeScale().setVisibleLogicalRange(range);
      syncing = false;
    };

    rsiChart?.timeScale().subscribeVisibleLogicalRangeChange(subSync);
    macdChart?.timeScale().subscribeVisibleLogicalRangeChange(subSync);
  }

  // ── Fetch and render data ────────────────────────────────────────────

  async function fetchAndRender() {
    loading = true;
    error = "";

    try {
      const resp = await fetch(
        `${API_BASE}/stock/${ticker}/chart?period=${period}`
      );

      if (!resp.ok) {
        const body = await resp.json().catch(() => ({}));
        throw new Error(body.detail || `HTTP ${resp.status}`);
      }

      const data = await resp.json();
      renderData(data);
    } catch (e: any) {
      error = e.message || "Failed to load chart data";
    } finally {
      loading = false;
    }
  }

  function renderData(data: any) {
    const { candles, indicators, quote } = data;

    if (!candleSeries || !volumeSeries) return;

    // Clear all indicator series first — prevents stale data when switching
    // to a shorter timeframe that doesn't include all indicators
    // (e.g., 1W has no MA200 because <200 data points).
    ma20Series?.setData([]);
    ma50Series?.setData([]);
    ma200Series?.setData([]);
    bbUpperSeries?.setData([]);
    bbMiddleSeries?.setData([]);
    bbLowerSeries?.setData([]);
    rsiSeries?.setData([]);
    macdLineSeries?.setData([]);
    macdSignalSeries?.setData([]);
    macdHistSeries?.setData([]);

    // Candlesticks
    candleSeries.setData(candles);

    // Current price — prefer live quote over last candle close
    if (quote?.price) {
      currentPrice = quote.price;
      priceChange = quote.change;
      priceChangePercent = quote.changePercent;
      priceDirection = priceChange >= 0 ? "up" : "down";
    } else if (candles.length >= 2) {
      const last = candles[candles.length - 1];
      const prev = candles[candles.length - 2];
      currentPrice = last.close;
      priceChange = last.close - prev.close;
      priceChangePercent = (priceChange / prev.close) * 100;
      priceDirection = priceChange >= 0 ? "up" : "down";
    } else if (candles.length === 1) {
      currentPrice = candles[0].close;
      priceChange = 0;
      priceChangePercent = 0;
      priceDirection = "up";
    }

    // Current price line — dashed horizontal line
    if (currentPrice > 0) {
      const priceColor = priceDirection === "up" ? "#26a69a" : "#ef5350";
      if ((candleSeries as any)._priceLine) {
        candleSeries.removePriceLine((candleSeries as any)._priceLine);
      }
      (candleSeries as any)._priceLine = candleSeries.createPriceLine({
        price: currentPrice,
        color: priceColor,
        lineWidth: 1,
        lineStyle: LineStyle.Dashed,
        axisLabelVisible: true,
        title: "",
      });
    }

    // Volume — colored by direction
    const volumeData = candles.map((c: any) => ({
      time: c.time,
      value: c.volume,
      color:
        c.close >= c.open
          ? "rgba(38, 166, 154, 0.35)"
          : "rgba(239, 83, 80, 0.35)",
    }));
    volumeSeries.setData(volumeData);

    // Moving averages — track which ones have data for the legend
    hasMA20 = !!indicators.ma20?.length;
    hasMA50 = !!indicators.ma50?.length;
    hasMA200 = !!indicators.ma200?.length;
    if (hasMA20) ma20Series?.setData(indicators.ma20);
    if (hasMA50) ma50Series?.setData(indicators.ma50);
    if (hasMA200) ma200Series?.setData(indicators.ma200);

    // Bollinger Bands
    hasBollinger = !!indicators.bollinger?.length;
    if (hasBollinger) {
      const bb = indicators.bollinger;
      bbUpperSeries?.setData(bb.map((p: any) => ({ time: p.time, value: p.upper })));
      bbMiddleSeries?.setData(bb.map((p: any) => ({ time: p.time, value: p.middle })));
      bbLowerSeries?.setData(bb.map((p: any) => ({ time: p.time, value: p.lower })));
    }

    // RSI
    if (indicators.rsi) {
      rsiSeries?.setData(indicators.rsi);

      const timeRange = indicators.rsi.map((p: any) => p.time);
      if (timeRange.length > 0 && rsiChart) {
        const refData = (val: number) =>
          timeRange.map((t: string) => ({ time: t, value: val }));
        (rsiChart as any)._overbought?.setData(refData(70));
        (rsiChart as any)._oversold?.setData(refData(30));
      }
    }

    // MACD
    if (indicators.macd) {
      const macd = indicators.macd;
      macdLineSeries?.setData(
        macd.map((p: any) => ({ time: p.time, value: p.macd }))
      );
      macdSignalSeries?.setData(
        macd.map((p: any) => ({ time: p.time, value: p.signal }))
      );
      macdHistSeries?.setData(
        macd.map((p: any) => ({
          time: p.time,
          value: p.histogram,
          color: p.histogram >= 0 ? "#26a69a" : "#ef5350",
        }))
      );
    }

    // Support/Resistance price lines
    // Remove previous S/R lines before adding new ones
    if ((candleSeries as any)._srLines) {
      for (const line of (candleSeries as any)._srLines) {
        candleSeries.removePriceLine(line);
      }
    }
    const srLines: any[] = [];
    const supportResistance = data.supportResistance || [];
    for (const level of supportResistance) {
      const isSupport = level.type === "support";
      const line = candleSeries.createPriceLine({
        price: level.price,
        color: isSupport ? "rgba(38, 166, 154, 0.6)" : "rgba(239, 83, 80, 0.6)",
        lineWidth: 1,
        lineStyle: LineStyle.Dotted,
        axisLabelVisible: false,
        title: "",
      });
      srLines.push(line);
    }
    (candleSeries as any)._srLines = srLines;

    // Pattern markers (arrows on the chart)
    const patternMarkers = (data.patterns || [])
      .filter((p: any) => p.time)
      .map((p: any) => ({
        time: p.time,
        position: p.direction === "bullish" ? "belowBar" : "aboveBar",
        color: p.direction === "bullish" ? "#26a69a" : "#ef5350",
        shape: p.direction === "bullish" ? "arrowUp" : "arrowDown",
        text: p.type.replace(/_/g, " ").substring(0, 12),
      }))
      .sort((a: any, b: any) => (a.time < b.time ? -1 : 1));

    if (patternMarkers.length > 0) {
      candleSeries.setMarkers(patternMarkers);
    } else {
      candleSeries.setMarkers([]);
    }

    // Show time-of-day on x-axis for intraday timeframes
    const intraday = period === "1w" || period === "1mo";
    const timeOpts = { timeVisible: intraday, secondsVisible: false };
    priceChart?.timeScale().applyOptions(timeOpts);
    rsiChart?.timeScale().applyOptions(timeOpts);
    macdChart?.timeScale().applyOptions(timeOpts);

    // Fit content
    priceChart?.timeScale().fitContent();
    rsiChart?.timeScale().fitContent();
    macdChart?.timeScale().fitContent();
  }

  // ── Toggle visibility ────────────────────────────────────────────────

  function updateVisibility() {
    ma20Series?.applyOptions({ visible: showMAs });
    ma50Series?.applyOptions({ visible: showMAs });
    ma200Series?.applyOptions({ visible: showMAs });

    bbUpperSeries?.applyOptions({ visible: showBollinger });
    bbMiddleSeries?.applyOptions({ visible: showBollinger });
    bbLowerSeries?.applyOptions({ visible: showBollinger });
  }

  function resizeAllCharts() {
    // After toggling sub-chart visibility the flex layout redistributes space.
    // Lightweight Charts won't pick that up automatically — we need to tell
    // each chart its container's new dimensions.
    if (priceContainer && priceChart) {
      priceChart.applyOptions({ width: priceContainer.clientWidth, height: priceContainer.clientHeight });
    }
    if (rsiContainer && rsiChart && showRSI) {
      rsiChart.applyOptions({ width: rsiContainer.clientWidth, height: rsiContainer.clientHeight });
    }
    if (macdContainer && macdChart && showMACD) {
      macdChart.applyOptions({ width: macdContainer.clientWidth, height: macdContainer.clientHeight });
    }
  }

  // ── Cleanup ──────────────────────────────────────────────────────────

  function cleanup() {
    resizeObserver?.disconnect();
    priceChart?.remove();
    rsiChart?.remove();
    macdChart?.remove();
    priceChart = null;
    rsiChart = null;
    macdChart = null;
  }

  // ── Lifecycle ────────────────────────────────────────────────────────

  onMount(() => {
    initCharts();
    fetchAndRender();

    return cleanup;
  });

  // Refetch when ticker or period changes
  $effect(() => {
    // Access reactive deps
    void ticker;
    void period;

    // Skip the initial mount (onMount handles it)
    if (priceChart && candleSeries) {
      fetchAndRender();
    }
  });

  // Update visibility when toggles change
  $effect(() => {
    void showMAs;
    void showBollinger;
    updateVisibility();
  });

  // Resize charts when RSI/MACD toggled — flex layout redistributes space
  // and Lightweight Charts needs to be told about the new dimensions.
  // requestAnimationFrame waits for the DOM to update after display:none changes.
  $effect(() => {
    void showRSI;
    void showMACD;
    requestAnimationFrame(() => resizeAllCharts());
  });
</script>

<div class="stock-chart">
  <ChartControls
    {period}
    onPeriodChange={(p) => (period = p)}
    {showMAs}
    onToggleMAs={() => (showMAs = !showMAs)}
    {showBollinger}
    onToggleBollinger={() => (showBollinger = !showBollinger)}
    {showRSI}
    onToggleRSI={() => (showRSI = !showRSI)}
    {showMACD}
    onToggleMACD={() => (showMACD = !showMACD)}
  />

  {#if currentPrice > 0}
    <div class="price-header">
      <span class="price-value">${currentPrice.toFixed(2)}</span>
      <span class="price-change" class:up={priceDirection === "up"} class:down={priceDirection === "down"}>
        {priceChange >= 0 ? "+" : ""}{priceChange.toFixed(2)} ({priceChange >= 0 ? "+" : ""}{priceChangePercent.toFixed(2)}%)
      </span>
    </div>
  {/if}

  <div class="chart-body">
    {#if loading}
      <div class="overlay">
        <span class="spinner"></span>
      </div>
    {/if}

    {#if error}
      <div class="error-message">
        <span>No data for {ticker}</span>
        <span class="error-detail">{error}</span>
      </div>
    {/if}

    <div class="price-chart" bind:this={priceContainer}></div>

    <!-- Use CSS display:none instead of {#if} to keep chart instances alive.
         Svelte {#if} destroys the DOM node, orphaning the chart instance. -->
    <div class="sub-chart-label" class:hidden={!showRSI}>RSI (14)</div>
    <div class="sub-chart" class:hidden={!showRSI} bind:this={rsiContainer}></div>

    <div class="sub-chart-label" class:hidden={!showMACD}>MACD (12, 26, 9)</div>
    <div class="sub-chart" class:hidden={!showMACD} bind:this={macdContainer}></div>
  </div>

  <div class="chart-legend">
    {#if showMAs && hasMA20}
      <span class="legend-item"><span class="dot" style="background:#2196F3"></span>MA20</span>
    {/if}
    {#if showMAs && hasMA50}
      <span class="legend-item"><span class="dot" style="background:#FF9800"></span>MA50</span>
    {/if}
    {#if showMAs && hasMA200}
      <span class="legend-item"><span class="dot" style="background:#F44336"></span>MA200</span>
    {/if}
    {#if showBollinger && hasBollinger}
      <span class="legend-item"><span class="dot" style="background:#00d4ff"></span>BB</span>
    {/if}
  </div>
</div>

<style>
  .stock-chart {
    display: flex;
    flex-direction: column;
    min-height: 100%;
    background: var(--bg-darker);
    border: 1px solid var(--border);
    border-radius: 6px;
  }

  .price-header {
    display: flex;
    align-items: baseline;
    gap: 0.75rem;
    padding: 0.4rem 0.75rem;
    border-bottom: 1px solid var(--border);
  }

  .price-value {
    font-size: 1.25rem;
    font-weight: 600;
    color: var(--text);
    font-variant-numeric: tabular-nums;
  }

  .price-change {
    font-size: 0.8rem;
    font-variant-numeric: tabular-nums;
  }

  .price-change.up {
    color: #26a69a;
  }

  .price-change.down {
    color: #ef5350;
  }

  .chart-body {
    position: relative;
    flex: 1;
    display: flex;
    flex-direction: column;
    min-height: 0;
  }

  .price-chart {
    flex: 1 1 0%;
    min-height: 150px;
  }

  .sub-chart {
    flex: 0 0 120px;
    min-height: 0;
    border-top: 1px solid var(--border);
  }

  .sub-chart-label {
    padding: 0.2rem 0.75rem;
    font-size: 0.65rem;
    color: var(--text-muted);
    background: var(--bg-darker);
    border-top: 1px solid var(--border);
    letter-spacing: 0.05em;
  }

  .hidden {
    display: none;
  }

  .overlay {
    position: absolute;
    inset: 0;
    display: flex;
    align-items: center;
    justify-content: center;
    background: rgba(7, 10, 26, 0.7);
    z-index: 10;
  }

  .spinner {
    width: 28px;
    height: 28px;
    border: 2px solid var(--border);
    border-top-color: var(--accent);
    border-radius: 50%;
    animation: spin 0.8s linear infinite;
  }

  @keyframes spin {
    to {
      transform: rotate(360deg);
    }
  }

  .error-message {
    position: absolute;
    inset: 0;
    display: flex;
    flex-direction: column;
    align-items: center;
    justify-content: center;
    gap: 0.5rem;
    color: var(--text-dim);
    font-size: 0.9rem;
    z-index: 5;
  }

  .error-detail {
    font-size: 0.75rem;
    color: var(--text-muted);
  }

  .chart-legend {
    display: flex;
    gap: 1rem;
    padding: 0.4rem 0.75rem;
    border-top: 1px solid var(--border);
    background: var(--bg-card);
    min-height: 1.5rem;
  }

  .legend-item {
    display: flex;
    align-items: center;
    gap: 0.3rem;
    font-size: 0.7rem;
    color: var(--text-dim);
  }

  .dot {
    width: 8px;
    height: 8px;
    border-radius: 2px;
    display: inline-block;
  }
</style>
