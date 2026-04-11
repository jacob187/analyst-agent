"use client";

import { useEffect, useRef } from "react";
import {
  createChart,
  CandlestickSeries,
  HistogramSeries,
  LineSeries,
  type IChartApi,
  type ISeriesApi,
  type CandlestickData,
  type Time,
  ColorType,
} from "lightweight-charts";
import type { ChartResponse } from "@/types";
import { PERIOD_MAP, type Period, type Indicator } from "@/lib/constants";

interface StockChartInnerProps {
  data: ChartResponse;
  period: Period;
  indicators: Set<Indicator>;
}

function resolveColor(cssVar: string, fallback: string): string {
  if (typeof window === "undefined") return fallback;
  const v = getComputedStyle(document.documentElement)
    .getPropertyValue(cssVar)
    .trim();
  return v ? `hsl(${v})` : fallback;
}

export default function StockChartInner({
  data,
  indicators,
}: StockChartInnerProps) {
  const containerRef = useRef<HTMLDivElement>(null);
  const rsiRef = useRef<HTMLDivElement>(null);
  const macdRef = useRef<HTMLDivElement>(null);

  const priceChartRef = useRef<IChartApi | null>(null);
  const rsiChartRef = useRef<IChartApi | null>(null);
  const macdChartRef = useRef<IChartApi | null>(null);

  useEffect(() => {
    if (!containerRef.current || !data.candles.length) return;

    const bg = resolveColor("--background", "#080c14");
    const cardBg = resolveColor("--card", "#0d1220");
    const borderColor = resolveColor("--border", "rgba(255,255,255,0.06)");
    const textColor = resolveColor("--muted-foreground", "#64748b");
    const emerald = "#10b981";
    const red = "#ef4444";

    const chartOpts = {
      layout: {
        background: { type: ColorType.Solid, color: cardBg },
        textColor,
      },
      grid: {
        vertLines: { color: borderColor },
        horzLines: { color: borderColor },
      },
      crosshair: { mode: 1 },
      rightPriceScale: { borderColor },
      timeScale: { borderColor, timeVisible: true },
    };

    // ── Price chart ──────────────────────────────────────────────────────────
    const priceChart = createChart(containerRef.current, {
      ...chartOpts,
      height: 340,
    });
    priceChartRef.current = priceChart;

    const candleSeries = priceChart.addSeries(CandlestickSeries, {
      upColor: emerald,
      downColor: red,
      borderUpColor: emerald,
      borderDownColor: red,
      wickUpColor: emerald,
      wickDownColor: red,
    });

    const candles = data.candles.map((c) => ({
      time: c.time as Time,
      open: c.open,
      high: c.high,
      low: c.low,
      close: c.close,
    })) as CandlestickData[];

    candleSeries.setData(candles);

    // Volume
    const volumeSeries = priceChart.addSeries(HistogramSeries, {
      color: emerald,
      priceFormat: { type: "volume" },
      priceScaleId: "volume",
    });
    priceChart.priceScale("volume").applyOptions({
      scaleMargins: { top: 0.8, bottom: 0 },
    });
    volumeSeries.setData(
      data.candles.map((c) => ({
        time: c.time as Time,
        value: c.volume,
        color: c.close >= c.open ? `${emerald}88` : `${red}88`,
      }))
    );

    // Moving averages
    const maColors: Record<string, string> = {
      ma20: "#60a5fa",
      ma50: "#f97316",
      ma200: "#ef4444",
    };
    if (indicators.has("MA")) {
      for (const key of ["ma20", "ma50", "ma200"] as const) {
        const raw = data.indicators[key] as
          | Array<{ time: Time; value: number }>
          | undefined;
        if (!raw?.length) continue;
        const s = priceChart.addSeries(LineSeries, {
          color: maColors[key],
          lineWidth: 1,
          lastValueVisible: false,
          priceLineVisible: false,
        });
        s.setData(raw);
      }
    }

    // Bollinger Bands
    if (indicators.has("BB")) {
      const bb = data.indicators.bollinger as
        | {
            upper: Array<{ time: Time; value: number }>;
            lower: Array<{ time: Time; value: number }>;
            middle: Array<{ time: Time; value: number }>;
          }
        | undefined;
      if (bb) {
        for (const band of [bb.upper, bb.middle, bb.lower]) {
          if (band?.length) {
            const s = priceChart.addSeries(LineSeries, {
              color: "#06b6d4",
              lineWidth: 1,
              lineStyle: 2, // dashed
              lastValueVisible: false,
              priceLineVisible: false,
            });
            s.setData(band);
          }
        }
      }
    }

    // Current price line
    const lastClose = data.candles[data.candles.length - 1]?.close;
    if (lastClose) {
      candleSeries.createPriceLine({
        price: lastClose,
        color: emerald,
        lineWidth: 1,
        lineStyle: 2,
        axisLabelVisible: true,
      });
    }

    priceChart.timeScale().fitContent();

    // ── RSI chart ────────────────────────────────────────────────────────────
    let rsiSeries: ISeriesApi<"Line"> | null = null;
    if (indicators.has("RSI") && rsiRef.current) {
      const rsiChart = createChart(rsiRef.current, {
        ...chartOpts,
        height: 120,
      });
      rsiChartRef.current = rsiChart;

      const rsiData = data.indicators.rsi as
        | Array<{ time: Time; value: number }>
        | undefined;
      if (rsiData?.length) {
        rsiSeries = rsiChart.addSeries(LineSeries, {
          color: "#a78bfa",
          lineWidth: 1,
          lastValueVisible: true,
          priceLineVisible: false,
        });
        rsiSeries.setData(rsiData);

        // Reference lines
        for (const level of [70, 30]) {
          rsiSeries.createPriceLine({
            price: level,
            color: level === 70 ? red : emerald,
            lineWidth: 1,
            lineStyle: 2,
            axisLabelVisible: true,
          });
        }
        rsiChart.timeScale().fitContent();
      }
    }

    // ── MACD chart ───────────────────────────────────────────────────────────
    if (indicators.has("MACD") && macdRef.current) {
      const macdChart = createChart(macdRef.current, {
        ...chartOpts,
        height: 120,
      });
      macdChartRef.current = macdChart;

      const macdData = data.indicators.macd as
        | {
            macd?: Array<{ time: Time; value: number }>;
            signal?: Array<{ time: Time; value: number }>;
            histogram?: Array<{ time: Time; value: number; color?: string }>;
          }
        | undefined;

      if (macdData) {
        if (macdData.histogram?.length) {
          const hist = macdChart.addSeries(HistogramSeries, {
            color: emerald,
            priceFormat: { type: "price", precision: 4 },
          });
          hist.setData(
            macdData.histogram.map((d) => ({
              time: d.time,
              value: d.value,
              color: d.value >= 0 ? `${emerald}cc` : `${red}cc`,
            }))
          );
        }
        if (macdData.macd?.length) {
          const macdLine = macdChart.addSeries(LineSeries, {
            color: "#60a5fa",
            lineWidth: 1,
            lastValueVisible: false,
            priceLineVisible: false,
          });
          macdLine.setData(macdData.macd);
        }
        if (macdData.signal?.length) {
          const signalLine = macdChart.addSeries(LineSeries, {
            color: "#f97316",
            lineWidth: 1,
            lastValueVisible: false,
            priceLineVisible: false,
          });
          signalLine.setData(macdData.signal);
        }
        macdChart.timeScale().fitContent();
      }
    }

    // ── Sync time scales ─────────────────────────────────────────────────────
    const charts = [priceChart, rsiChartRef.current, macdChartRef.current].filter(Boolean) as IChartApi[];
    charts.forEach((src) => {
      src.timeScale().subscribeVisibleLogicalRangeChange((range) => {
        if (!range) return;
        charts.forEach((tgt) => {
          if (tgt !== src) tgt.timeScale().setVisibleLogicalRange(range);
        });
      });
    });

    // ── Resize observer ──────────────────────────────────────────────────────
    const ro = new ResizeObserver(() => {
      if (containerRef.current)
        priceChart.applyOptions({ width: containerRef.current.clientWidth });
    });
    if (containerRef.current) ro.observe(containerRef.current);

    return () => {
      ro.disconnect();
      priceChart.remove();
      rsiChartRef.current?.remove();
      macdChartRef.current?.remove();
    };
  // eslint-disable-next-line react-hooks/exhaustive-deps
  }, [data, indicators]);

  const showRsi = indicators.has("RSI");
  const showMacd = indicators.has("MACD");

  return (
    <div className="w-full space-y-px rounded-xl overflow-hidden border border-border/60">
      <div ref={containerRef} className="w-full" />
      {showRsi && (
        <div className="border-t border-border/40">
          <p className="px-3 pt-1.5 text-[10px] font-medium uppercase tracking-widest text-muted-foreground">
            RSI (14)
          </p>
          <div ref={rsiRef} className="w-full" />
        </div>
      )}
      {showMacd && (
        <div className="border-t border-border/40">
          <p className="px-3 pt-1.5 text-[10px] font-medium uppercase tracking-widest text-muted-foreground">
            MACD (12, 26, 9)
          </p>
          <div ref={macdRef} className="w-full" />
        </div>
      )}
    </div>
  );
}
