/**
 * CandlestickChart — TradingView Lightweight Charts wrapper.
 *
 * Renders OHLCV candlesticks from BarData (market.py history endpoint).
 * Forecast dots are drawn as coloured markers on a separate line series.
 */
import React, { useEffect, useRef } from 'react';
import {
  createChart,
  CandlestickSeries,
  LineSeries,
  ColorType,
  LineStyle,
  createSeriesMarkers,
} from 'lightweight-charts';
import type { BarData } from '../services/marketService';

export interface ForecastDot {
  /** ISO date string "YYYY-MM-DD" */
  date: string;
  price: number;
  label: string;  // "+1d", "+3d", "+7d"
}

interface CandlestickChartProps {
  bars: BarData[];
  forecastDots?: ForecastDot[];
  market: 'VN' | 'NASDAQ';
  height?: number;
}

export const CandlestickChart: React.FC<CandlestickChartProps> = ({
  bars,
  forecastDots = [],
  market,
  height = 380,
}) => {
  const containerRef = useRef<HTMLDivElement>(null);

  useEffect(() => {
    if (!containerRef.current || !bars.length) return;

    // ── Chart creation ────────────────────────────────────────────────────────
    const chart = createChart(containerRef.current, {
      width:  containerRef.current.clientWidth,
      height,
      layout: {
        background: { type: ColorType.Solid, color: 'transparent' },
        textColor: '#94A3B8',
        fontFamily: "'JetBrains Mono', 'Fira Code', monospace",
        fontSize: 10,
      },
      grid: {
        vertLines:  { color: '#F1F5F9', style: LineStyle.Dashed },
        horzLines:  { color: '#F1F5F9', style: LineStyle.Dashed },
      },
      crosshair: {
        vertLine: { color: '#4F46E5', width: 1, style: LineStyle.Dashed, labelBackgroundColor: '#4F46E5' },
        horzLine: { color: '#4F46E5', width: 1, style: LineStyle.Dashed, labelBackgroundColor: '#4F46E5' },
      },
      rightPriceScale: {
        borderVisible: false,
        scaleMargins: { top: 0.1, bottom: 0.15 },
        ticksVisible: true,
      },
      timeScale: {
        borderVisible: false,
        fixLeftEdge: true,
        fixRightEdge: true,
        tickMarkFormatter: (time: number) => {
          const d = new Date(time * 1000);
          return d.toLocaleDateString('en-US', { month: 'short', day: 'numeric' });
        },
      },
      handleScroll: true,
      handleScale:  true,
    });

    // ── Resize observer ───────────────────────────────────────────────────────
    const ro = new ResizeObserver(() => {
      if (containerRef.current) {
        chart.applyOptions({ width: containerRef.current.clientWidth });
      }
    });
    ro.observe(containerRef.current);

    // ── Candlestick series ────────────────────────────────────────────────────
    const candleSeries = chart.addSeries(CandlestickSeries, {
      upColor:          '#10B981',
      downColor:        '#EF4444',
      borderUpColor:    '#10B981',
      borderDownColor:  '#EF4444',
      wickUpColor:      '#10B981',
      wickDownColor:    '#EF4444',
    });

    // Map BarData → lightweight-charts candle format (time = Unix seconds)
    const candleData = bars
      .filter(b => b.date_iso && b.open != null && b.high != null && b.low != null && b.close != null)
      .map(b => ({
        time:  Math.floor(new Date(b.date_iso!).getTime() / 1000) as unknown as import('lightweight-charts').Time,
        open:  b.open!,
        high:  b.high!,
        low:   b.low!,
        close: b.close!,
      }))
      .sort((a, b) => (a.time as number) - (b.time as number));

    candleSeries.setData(candleData);

    // ── Forecast line + markers ───────────────────────────────────────────────
    if (forecastDots.length > 0 && candleData.length > 0) {
      const lastCandle = candleData[candleData.length - 1];
      const lastClose  = bars[bars.length - 1]?.close ?? 0;

      // Build one line series: bridge (lastClose) → each forecast dot
      const forecastLineSeries = chart.addSeries(LineSeries, {
        color:       '#10B981',
        lineWidth:   2,
        lineStyle:   LineStyle.Dashed,
        pointMarkersVisible: false,
        lastValueVisible:    false,
        priceLineVisible:    false,
      });

      const linePoints = [
        { time: lastCandle.time, value: lastClose },
        ...forecastDots.map(d => ({
          time:  Math.floor(new Date(d.date).getTime() / 1000) as unknown as import('lightweight-charts').Time,
          value: d.price,
        })),
      ].sort((a, b) => (a.time as number) - (b.time as number));

      forecastLineSeries.setData(linePoints);

      // Markers at each forecast dot (v5 API)
      createSeriesMarkers(
        forecastLineSeries,
        forecastDots.map(d => ({
          time:     Math.floor(new Date(d.date).getTime() / 1000) as unknown as import('lightweight-charts').Time,
          position: 'aboveBar' as const,
          color:    '#10B981',
          shape:    'circle' as const,
          text:     d.label,
          size:     1,
        }))
      );
    }

    // ── Price formatter ───────────────────────────────────────────────────────
    chart.applyOptions({
      localization: {
        priceFormatter: (price: number) =>
          market === 'VN'
            ? (price / 1000).toFixed(1) + 'k'
            : price.toFixed(2),
      },
    });

    chart.timeScale().fitContent();

    return () => {
      ro.disconnect();
      chart.remove();
    };
  }, [bars, forecastDots, market, height]);

  return (
    <div ref={containerRef} style={{ width: '100%', height }} />
  );
};
