import React, { useState, useEffect, useMemo } from 'react';
import {
  ResponsiveContainer, AreaChart, Area,
  XAxis, YAxis, Tooltip, CartesianGrid, ReferenceLine,
} from 'recharts';
import { fetchStockHistory, fetchPrediction, fetchNasdaqPrediction, BarData } from '../services/marketService';
import { motion, AnimatePresence } from 'motion/react';
import { TrendingUp, TrendingDown, Loader2 } from 'lucide-react';
import { cn } from '../lib/utils';
import { CandlestickChart, ForecastDot } from './CandlestickChart';

interface PricePredictionsProps {
  ticker: string;
  currentPrice: number;
  market: 'NASDAQ' | 'VN';
}

const DAY_KEY: Record<string, string> = { '1d': 'day_1', '3d': 'day_3', '7d': 'day_7' };
const HORIZON_DAYS: Record<string, number> = { '1d': 1, '3d': 3, '7d': 7 };

const WEEKDAYS = ['Sunday','Monday','Tuesday','Wednesday','Thursday','Friday','Saturday'];
const MONTHS   = ['Jan','Feb','Mar','Apr','May','Jun','Jul','Aug','Sep','Oct','Nov','Dec'];

/** Return the date N trading days from today, formatted as "Monday 28 Apr 2026". */
function tradingDateLabel(n: number): string {
  const date = new Date();
  let count = 0;
  while (count < n) {
    date.setDate(date.getDate() + 1);
    const dow = date.getDay();
    if (dow !== 0 && dow !== 6) count++;
  }
  return `${WEEKDAYS[date.getDay()]} ${date.getDate()} ${MONTHS[date.getMonth()]} ${date.getFullYear()}`;
}

type ChartPoint = { date: string; price: number | null; forecast: number | null; isFuture?: boolean };

function buildChartData(bars: BarData[], predictions: Record<string, number>, horizon: '1d' | '3d' | '7d'): ChartPoint[] {
  const historical: ChartPoint[] = bars.slice(-40).map(b => ({
    date:     b.date,
    price:    b.close ?? null,
    forecast: null,
  }));

  const lastClose = bars[bars.length - 1]?.close ?? 0;
  const nDays = HORIZON_DAYS[horizon];

  // Build future segment: interpolate linearly from lastClose → predicted
  const targetPrice = predictions[DAY_KEY[horizon]];
  if (!targetPrice || !lastClose) return historical;

  const futurePoints: ChartPoint[] = [];
  for (let d = 1; d <= nDays; d++) {
    const t = d / nDays;
    futurePoints.push({
      date:     `+${d}d`,
      price:    null,
      forecast: parseFloat((lastClose + (targetPrice - lastClose) * t).toFixed(2)),
      isFuture: true,
    });
  }

  // Bridge: last historical point doubles as start of forecast line
  const bridgePoint: ChartPoint = {
    date:     historical[historical.length - 1]?.date ?? 'Today',
    price:    lastClose,
    forecast: lastClose,
    isFuture: false,
  };

  // Replace last historical with bridge
  return [...historical.slice(0, -1), bridgePoint, ...futurePoints];
}

const CustomTooltip = ({ active, payload, market }: any) => {
  if (!active || !payload?.length) return null;
  const d = payload[0].payload as ChartPoint;
  const val = d.forecast ?? d.price;
  if (val == null) return null;
  const fmt = (v: number) => market === 'VN' ? v.toLocaleString('vi-VN') : v.toLocaleString(undefined, { minimumFractionDigits: 2 });
  return (
    <div className="bg-bg-surface border border-border-theme px-4 py-3 rounded-xl shadow-xl text-xs font-mono">
      <p className="text-text-muted mb-1 font-sans font-bold uppercase tracking-wider">{d.date}</p>
      <p className="text-text-primary font-bold text-sm">{fmt(val)}</p>
      {d.isFuture && <p className="text-accent-theme text-[10px] font-sans font-bold mt-1 uppercase tracking-wider">Forecast</p>}
    </div>
  );
};

export const PricePredictions: React.FC<PricePredictionsProps> = ({ ticker, currentPrice, market }) => {
  const [horizon, setHorizon]         = useState<'1d' | '3d' | '7d'>('7d');
  const [chartType, setChartType]     = useState<'area' | 'candle'>('area');
  const effectiveHorizon = horizon;
  const [bars, setBars]               = useState<BarData[]>([]);
  const [predictions, setPredictions] = useState<Record<string, number>>({});
  const [loading, setLoading]         = useState(true);
  const [range, setRange]             = useState<'1mo' | '3mo'>('1mo');

  // Load history + predictions for both markets
  useEffect(() => {
    setLoading(true);
    setPredictions({});

    const histFetch = fetchStockHistory(ticker, range, market);

    let predFetch: Promise<any>;
    if (market === 'VN') {
      // Task 2: three horizons (1d, 3d, 7d).
      // Each model returns output_size=1 with key 'day_1', but conceptually
      // task2_1 → day_1, task2_2_k3 → day_3, task2_2_k7 → day_7.
      // Remap explicitly so the merged object has all three keys.
      predFetch = Promise.all([
        fetchPrediction(ticker, 'task2_1').catch(() => null),
        fetchPrediction(ticker, 'task2_2_k3').catch(() => null),
        fetchPrediction(ticker, 'task2_2_k7').catch(() => null),
      ]).then(([r1, r3, r7]) => {
        const merged: Record<string, number> = {};
        const firstVal = (r: any) => r ? Object.values(r.predictions as Record<string, number>)[0] : undefined;
        const v1 = firstVal(r1); if (v1 !== undefined) merged['day_1'] = v1;
        const v3 = firstVal(r3); if (v3 !== undefined) merged['day_3'] = v3;
        const v7 = firstVal(r7); if (v7 !== undefined) merged['day_7'] = v7;
        return merged;
      });
    } else {
      // Task 1: three horizons — k=1 (day_1), k=3 (day_3), k=7 (day_7)
      predFetch = Promise.all([
        fetchNasdaqPrediction(ticker, 1).catch(() => null),
        fetchNasdaqPrediction(ticker, 3).catch(() => null),
        fetchNasdaqPrediction(ticker, 7).catch(() => null),
      ]).then(([r1, r3, r7]) => {
        const merged: Record<string, number> = {};
        if (r1?.predictions?.day_1 !== undefined) merged['day_1'] = r1.predictions.day_1;
        if (r3?.predictions?.day_3 !== undefined) merged['day_3'] = r3.predictions.day_3;
        if (r7?.predictions?.day_7 !== undefined) merged['day_7'] = r7.predictions.day_7;
        return merged;
      });
    }

    Promise.all([histFetch, predFetch])
      .then(([histBars, preds]) => {
        setBars(histBars as BarData[]);
        setPredictions(preds as Record<string, number>);
      })
      .catch(console.error)
      .finally(() => setLoading(false));
  }, [ticker, market, range]);

  const chartData = useMemo(
    () => buildChartData(bars, predictions, effectiveHorizon),
    [bars, predictions, effectiveHorizon]
  );

  const lastClose     = bars[bars.length - 1]?.close ?? currentPrice;
  const predPrice     = predictions[DAY_KEY[effectiveHorizon]];
  const priceDiff     = predPrice && lastClose ? predPrice - lastClose : 0;
  const pricePercent  = lastClose ? (priceDiff / lastClose) * 100 : 0;
  const prevClose     = bars[bars.length - 2]?.close;
  const dayChange     = lastClose && prevClose ? ((lastClose - prevClose) / prevClose) * 100 : 0;

  const fmtPrice = (v: number) =>
    market === 'VN' ? v.toLocaleString('vi-VN') : v.toLocaleString(undefined, { minimumFractionDigits: 2 });

  const yDomain = useMemo(() => {
    const vals = chartData.flatMap(d => [d.price, d.forecast]).filter(Boolean) as number[];
    if (!vals.length) return ['auto' as const, 'auto' as const];
    const min = Math.min(...vals);
    const max = Math.max(...vals);
    const pad = (max - min) * 0.08;
    return [min - pad, max + pad] as [number, number];
  }, [chartData]);

  if (loading) {
    return (
      <div className="flex flex-col items-center justify-center h-[600px] text-text-secondary">
        <Loader2 className="animate-spin mb-4 text-accent-theme" size={40} />
        <p className="font-mono text-xs uppercase tracking-widest">Loading Market Data…</p>
      </div>
    );
  }

  return (
    <div className="space-y-6 pb-20">

      {/* ── Top metrics row ───────────────────────────────────────────────── */}
      <div className="grid grid-cols-4 gap-4">
        {[
          {
            label: 'Last Close',
            value: lastClose ? fmtPrice(lastClose) : '—',
            sub: market === 'VN' ? 'VND' : 'USD',
            color: 'text-text-primary',
          },
          {
            label: 'Day Change',
            value: prevClose ? `${dayChange >= 0 ? '+' : ''}${dayChange.toFixed(2)}%` : '—',
            sub: 'vs yesterday',
            color: dayChange >= 0 ? 'text-pos' : 'text-neg',
          },
          {
            label: `Forecast — ${tradingDateLabel(HORIZON_DAYS[effectiveHorizon])}`,
            value: predPrice ? fmtPrice(predPrice) : '—',
            sub: market === 'VN' ? 'CNN-LSTM target' : 'Task 1 LSTM • next day',
            color: 'text-accent-theme',
          },
          {
            label: 'Expected Return',
            value: predPrice ? `${pricePercent >= 0 ? '+' : ''}${pricePercent.toFixed(2)}%` : '—',
            sub: `by ${tradingDateLabel(HORIZON_DAYS[effectiveHorizon])}`,
            color: priceDiff >= 0 ? 'text-pos' : 'text-neg',
          },
        ].map((m) => (
          <div key={m.label} className="theme-card p-5">
            <p className="theme-label mb-1">{m.label}</p>
            <p className={cn("text-xl font-mono font-bold tabular-nums tracking-tighter", m.color)}>{m.value}</p>
            <p className="text-[10px] text-text-muted font-bold mt-1 uppercase tracking-wider">{m.sub}</p>
          </div>
        ))}
      </div>

      {/* ── Main chart card ───────────────────────────────────────────────── */}
      <div className="theme-card p-8">
        {/* Chart header */}
        <div className="flex items-center justify-between mb-6">
          <div>
            <h3 className="theme-label mb-1">Price Trajectory + CNN-LSTM Forecast</h3>
            <div className="flex items-center gap-4 text-sm">
              <span className="flex items-center gap-1.5">
                <span className="w-8 h-0.5 bg-accent-theme inline-block rounded" />
                <span className="text-text-muted font-medium text-xs">Historical</span>
              </span>
              <span className="flex items-center gap-1.5">
                <span className="w-8 h-0.5 border-t-2 border-dashed border-pos inline-block" />
                <span className="text-text-muted font-medium text-xs">Forecast</span>
              </span>
            </div>
          </div>

          <div className="flex items-center gap-4">
            {/* Chart type toggle */}
            <div className="bg-bg-deep p-1 rounded-xl flex gap-1 border border-border-theme/60">
              {(['area', 'candle'] as const).map((ct) => (
                <button key={ct} onClick={() => setChartType(ct)}
                  className={cn(
                    "px-3 py-1.5 text-[10px] font-black uppercase tracking-[0.15em] rounded-lg transition-all",
                    chartType === ct ? "bg-white text-accent-theme shadow-sm border border-border-theme" : "text-text-muted hover:text-text-secondary"
                  )}>
                  {ct === 'area' ? 'Area' : 'Candle'}
                </button>
              ))}
            </div>
            {/* Horizon */}
            <div className="bg-bg-deep p-1 rounded-xl flex gap-1 border border-border-theme/60">
              {(['1d', '3d', '7d'] as const).map((h) => {
                return (
                  <button key={h}
                    onClick={() => setHorizon(h)}
                    className={cn(
                      "px-4 py-1.5 text-[10px] font-black uppercase tracking-[0.15em] rounded-lg transition-all",
                      effectiveHorizon === h ? "bg-white text-accent-theme shadow-sm border border-border-theme" : "text-text-muted hover:text-text-secondary",
                      false
                    )}>
                    {h}
                  </button>
                );
              })}
            </div>
            {/* Range */}
            <div className="flex gap-1">
              {(['1mo', '3mo'] as const).map((r) => (
                <button key={r} onClick={() => setRange(r)}
                  className={cn(
                    "px-3 py-1.5 text-[10px] font-bold uppercase tracking-widest rounded-lg border transition-all",
                    range === r ? "bg-white text-accent-theme border-border-theme shadow-sm" : "border-transparent text-text-muted hover:text-text-secondary"
                  )}>
                  {r.replace('mo', 'M')}
                </button>
              ))}
            </div>
          </div>
        </div>

        {/* Chart */}
        <div className="h-[380px]">
          {chartType === 'candle' ? (
            <CandlestickChart
              bars={bars}
              market={market}
              height={380}
              forecastDots={(() => {
                const dots: ForecastDot[] = [];
                const horizons: Array<{ key: string; days: number; label: string }> = [
                  { key: 'day_1', days: 1, label: '+1d' },
                  { key: 'day_3', days: 3, label: '+3d' },
                  { key: 'day_7', days: 7, label: '+7d' },
                ];
                const lastIso = bars[bars.length - 1]?.date_iso;
                if (!lastIso) return dots;
                for (const h of horizons) {
                  const price = predictions[h.key];
                  if (!price) continue;
                  // Advance N trading days from last bar date
                  const d = new Date(lastIso);
                  let count = 0;
                  while (count < h.days) {
                    d.setDate(d.getDate() + 1);
                    if (d.getDay() !== 0 && d.getDay() !== 6) count++;
                  }
                  dots.push({
                    date:  d.toISOString().slice(0, 10),
                    price,
                    label: h.label,
                  });
                }
                return dots;
              })()}
            />
          ) : (
          <ResponsiveContainer width="100%" height="100%">
            <AreaChart data={chartData} margin={{ top: 10, right: 60, bottom: 0, left: 0 }}>
              <defs>
                <linearGradient id="gradHist" x1="0" y1="0" x2="0" y2="1">
                  <stop offset="0%"   stopColor="#4F46E5" stopOpacity={0.15} />
                  <stop offset="100%" stopColor="#4F46E5" stopOpacity={0} />
                </linearGradient>
                <linearGradient id="gradFore" x1="0" y1="0" x2="0" y2="1">
                  <stop offset="0%"   stopColor="#10B981" stopOpacity={0.2} />
                  <stop offset="100%" stopColor="#10B981" stopOpacity={0} />
                </linearGradient>
              </defs>
              <CartesianGrid strokeDasharray="3 3" stroke="#F1F5F9" vertical={false} />
              <XAxis
                dataKey="date"
                axisLine={false} tickLine={false}
                tick={{ fill: '#94A3B8', fontSize: 10, fontWeight: 700 }}
                minTickGap={35} dy={12}
              />
              <YAxis
                domain={yDomain}
                axisLine={false} tickLine={false}
                tick={{ fill: '#94A3B8', fontSize: 10, fontWeight: 700 }}
                orientation="right" dx={12}
                tickFormatter={(v) => market === 'VN' ? (v / 1000).toFixed(0) + 'k' : v.toFixed(0)}
              />
              <Tooltip content={<CustomTooltip market={market} />}
                cursor={{ stroke: '#4F46E5', strokeWidth: 1, strokeDasharray: '4 4' }} />

              {/* Historical price area */}
              <Area
                type="monotone" dataKey="price"
                stroke="#4F46E5" strokeWidth={2}
                fill="url(#gradHist)"
                dot={false} activeDot={{ r: 4, fill: '#4F46E5', strokeWidth: 0 }}
                connectNulls={false}
              />

              {/* Forecast area */}
              <Area
                type="monotone" dataKey="forecast"
                stroke="#10B981" strokeWidth={2.5} strokeDasharray="6 4"
                fill="url(#gradFore)"
                dot={false} activeDot={{ r: 5, fill: '#10B981', strokeWidth: 0 }}
                connectNulls={false}
              />

              {/* Forecast target line */}
              {predPrice && (
                <ReferenceLine
                  y={predPrice} stroke="#10B981" strokeDasharray="4 4" strokeOpacity={0.5}
                  label={{ value: `▶ ${fmtPrice(predPrice)}`, position: 'right', fill: '#10B981', fontSize: 10, fontWeight: 800 }}
                />
              )}
            </AreaChart>
          </ResponsiveContainer>
          )}
        </div>

        {/* Prediction summary banner */}
        {predPrice && market === 'VN' && (
          <AnimatePresence mode="wait">
            <motion.div
              key={horizon}
              initial={{ opacity: 0, y: 6 }}
              animate={{ opacity: 1, y: 0 }}
              transition={{ duration: 0.3 }}
              className={cn(
                "mt-6 flex items-center justify-between px-6 py-4 rounded-2xl border",
                priceDiff >= 0 ? "bg-pos/5 border-pos/20" : "bg-neg/5 border-neg/20"
              )}
            >
              <div className="flex items-center gap-3">
                {priceDiff >= 0
                  ? <TrendingUp size={20} className="text-pos" />
                  : <TrendingDown size={20} className="text-neg" />}
                <div>
                  <p className="text-xs font-black uppercase tracking-widest text-text-muted">CNN-LSTM Forecast</p>
                  <p className={cn("text-sm font-bold", priceDiff >= 0 ? "text-pos" : "text-neg")}>
                    {priceDiff >= 0 ? 'Upward' : 'Downward'} target by <span className="font-mono">{tradingDateLabel(HORIZON_DAYS[effectiveHorizon])}</span>
                    {' — '}
                    <span className="font-mono">{fmtPrice(predPrice)}</span>
                    {' '}({pricePercent >= 0 ? '+' : ''}{pricePercent.toFixed(2)}%)
                  </p>
                </div>
              </div>
              <div className="text-right">
                <p className="text-[9px] text-text-muted font-bold uppercase tracking-widest">from last close</p>
                <p className="font-mono text-sm font-bold text-text-primary">{fmtPrice(lastClose)}</p>
              </div>
            </motion.div>
          </AnimatePresence>
        )}
      </div>

    </div>
  );
};
