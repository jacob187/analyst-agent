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

  function chartOptions(height: number) {
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
    });

    volumeSeries = priceChart.addSeries(HistogramSeries, {
      priceFormat: { type: "volume" },
      priceScaleId: "volume",
    });
    priceChart.priceScale("volume").applyOptions({
      scaleMargins: { top: 0.85, bottom: 0 },
    });

    // MA overlays
    ma20Series = priceChart.addSeries(LineSeries, {
      color: "#2196F3",
      lineWidth: 1,
      title: "MA20",
      visible: showMAs,
    });
    ma50Series = priceChart.addSeries(LineSeries, {
      color: "#FF9800",
      lineWidth: 1,
      title: "MA50",
      visible: showMAs,
    });
    ma200Series = priceChart.addSeries(LineSeries, {
      color: "#F44336",
      lineWidth: 1,
      title: "MA200",
      visible: showMAs,
    });

    // Bollinger Bands
    bbUpperSeries = priceChart.addSeries(LineSeries, {
      color: "rgba(0, 212, 255, 0.3)",
      lineWidth: 1,
      lineStyle: LineStyle.Dashed,
      visible: showBollinger,
    });
    bbMiddleSeries = priceChart.addSeries(LineSeries, {
      color: "rgba(0, 212, 255, 0.15)",
      lineWidth: 1,
      visible: showBollinger,
    });
    bbLowerSeries = priceChart.addSeries(LineSeries, {
      color: "rgba(0, 212, 255, 0.3)",
      lineWidth: 1,
      lineStyle: LineStyle.Dashed,
      visible: showBollinger,
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

      macdHistSeries = macdChart.addSeries(HistogramSeries);
      macdLineSeries = macdChart.addSeries(LineSeries, {
        color: "#2962FF",
        lineWidth: 1,
        title: "MACD",
      });
      macdSignalSeries = macdChart.addSeries(LineSeries, {
        color: "#FF6D00",
        lineWidth: 1,
        title: "Signal",
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

  function syncCharts() {
    if (!priceChart) return;

    priceChart
      .timeScale()
      .subscribeVisibleLogicalRangeChange((range) => {
        if (range) {
          if (rsiChart) rsiChart.timeScale().setVisibleLogicalRange(range);
          if (macdChart) macdChart.timeScale().setVisibleLogicalRange(range);
        }
      });

    // Sync scrolling from sub-charts back to price chart
    if (rsiChart) {
      rsiChart
        .timeScale()
        .subscribeVisibleLogicalRangeChange((range) => {
          if (range && priceChart)
            priceChart.timeScale().setVisibleLogicalRange(range);
          if (range && macdChart)
            macdChart.timeScale().setVisibleLogicalRange(range);
        });
    }

    if (macdChart) {
      macdChart
        .timeScale()
        .subscribeVisibleLogicalRangeChange((range) => {
          if (range && priceChart)
            priceChart.timeScale().setVisibleLogicalRange(range);
          if (range && rsiChart)
            rsiChart.timeScale().setVisibleLogicalRange(range);
        });
    }
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
    const { candles, indicators } = data;

    if (!candleSeries || !volumeSeries) return;

    // Candlesticks
    candleSeries.setData(candles);

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

    // Moving averages
    if (indicators.ma20 && ma20Series) ma20Series.setData(indicators.ma20);
    if (indicators.ma50 && ma50Series) ma50Series.setData(indicators.ma50);
    if (indicators.ma200 && ma200Series) ma200Series.setData(indicators.ma200);

    // Bollinger Bands
    if (indicators.bollinger) {
      const bb = indicators.bollinger;
      if (bbUpperSeries) bbUpperSeries.setData(bb.map((p: any) => ({ time: p.time, value: p.upper })));
      if (bbMiddleSeries) bbMiddleSeries.setData(bb.map((p: any) => ({ time: p.time, value: p.middle })));
      if (bbLowerSeries) bbLowerSeries.setData(bb.map((p: any) => ({ time: p.time, value: p.lower })));
    }

    // RSI
    if (indicators.rsi && rsiSeries) {
      rsiSeries.setData(indicators.rsi);

      // Set overbought/oversold reference lines across the time range
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
      if (macdLineSeries)
        macdLineSeries.setData(
          macd.map((p: any) => ({ time: p.time, value: p.macd }))
        );
      if (macdSignalSeries)
        macdSignalSeries.setData(
          macd.map((p: any) => ({ time: p.time, value: p.signal }))
        );
      if (macdHistSeries)
        macdHistSeries.setData(
          macd.map((p: any) => ({
            time: p.time,
            value: p.histogram,
            color: p.histogram >= 0 ? "#26a69a" : "#ef5350",
          }))
        );
    }

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

    {#if showRSI}
      <div class="sub-chart-label">RSI (14)</div>
      <div class="sub-chart" bind:this={rsiContainer}></div>
    {/if}

    {#if showMACD}
      <div class="sub-chart-label">MACD (12, 26, 9)</div>
      <div class="sub-chart" bind:this={macdContainer}></div>
    {/if}
  </div>

  <div class="chart-legend">
    {#if showMAs}
      <span class="legend-item"><span class="dot" style="background:#2196F3"></span>MA20</span>
      <span class="legend-item"><span class="dot" style="background:#FF9800"></span>MA50</span>
      <span class="legend-item"><span class="dot" style="background:#F44336"></span>MA200</span>
    {/if}
    {#if showBollinger}
      <span class="legend-item"><span class="dot" style="background:#00d4ff"></span>BB</span>
    {/if}
  </div>
</div>

<style>
  .stock-chart {
    display: flex;
    flex-direction: column;
    height: 100%;
    min-height: 400px;
    background: var(--bg-darker);
    border: 1px solid var(--border);
    border-radius: 6px;
    overflow: hidden;
  }

  .chart-body {
    position: relative;
    flex: 1;
    display: flex;
    flex-direction: column;
  }

  .price-chart {
    flex: 1;
    min-height: 300px;
  }

  .sub-chart {
    height: 120px;
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
