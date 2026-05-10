import React, { useState, useEffect, useMemo } from 'react';
import {
  ResponsiveContainer, AreaChart, Area,
  XAxis, YAxis, Tooltip, CartesianGrid, ReferenceLine,
} from 'recharts';
import {
  fetchStockHistory, fetchPrediction,
  fetchNasdaqPrediction, fetchNasdaqConsecutivePrediction,
  BarData,
} from '../services/marketService';
import { motion, AnimatePresence } from 'motion/react';
import { TrendingUp, TrendingDown, Loader2 } from 'lucide-react';
import { cn } from '../lib/utils';
import { CandlestickChart, ForecastDot } from './CandlestickChart';

interface PricePredictionsProps {
  ticker: string;
  currentPrice: number;
  market: 'NASDAQ' | 'VN';
}

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

// Keyed by horizon — each holds the model's per-day predictions for that horizon
type HorizonPredictions = Record<'1d' | '3d' | '7d', Record<string, number>>;

/**
 * Build area-chart data from OHLCV bars + the selected horizon's predictions.
 * Future points come from the model's actual per-day outputs (day_1 … day_k),
 * NOT from linear interpolation.
 */
function buildChartData(
  bars: BarData[],
  horizonPreds: Record<string, number>,
  horizon: '1d' | '3d' | '7d',
): ChartPoint[] {
  const historical: ChartPoint[] = bars.slice(-40).map(b => ({
    date:     b.date,
    price:    b.close ?? null,
    forecast: null,
  }));

  const lastClose = bars[bars.length - 1]?.close ?? 0;
  const nDays     = HORIZON_DAYS[horizon];

  if (!lastClose) return historical;

  // Plot actual model predictions for each future day
  const futurePoints: ChartPoint[] = [];
  for (let d = 1; d <= nDays; d++) {
    const dayPrice = horizonPreds[`day_${d}`];
    if (dayPrice === undefined) break;
    futurePoints.push({
      date:     `+${d}d`,
      price:    null,
      forecast: parseFloat(dayPrice.toFixed(2)),
      isFuture: true,
    });
  }

  if (!futurePoints.length) return historical;

  // Bridge: last historical bar doubles as forecast line start
  const bridgePoint: ChartPoint = {
    date:     historical[historical.length - 1]?.date ?? 'Today',
    price:    lastClose,
    forecast: lastClose,
    isFuture: false,
  };

  return [...historical.slice(0, -1), bridgePoint, ...futurePoints];
}

const CustomTooltip = ({ active, payload, market }: any) => {
  if (!active || !payload?.length) return null;
  const d = payload[0].payload as ChartPoint;
  const val = d.forecast ?? d.price;
  if (val == null) return null;
  const fmt = (v: number) =>
    market === 'VN'
      ? v.toLocaleString('vi-VN')
      : v.toLocaleString(undefined, { minimumFractionDigits: 2 });
  return (
    <div className="bg-bg-surface border border-border-theme px-4 py-3 rounded-xl shadow-xl text-xs font-mono">
      <p className="text-text-muted mb-1 font-sans font-bold uppercase tracking-wider">{d.date}</p>
      <p className="text-text-primary font-bold text-sm">{fmt(val)}</p>
      {d.isFuture && (
        <p className="text-accent-theme text-[10px] font-sans font-bold mt-1 uppercase tracking-wider">
          Forecast
        </p>
      )}
    </div>
  );
};

export const PricePredictions: React.FC<PricePredictionsProps> = ({ ticker, currentPrice, market }) => {
  const [horizon, setHorizon]     = useState<'1d' | '3d' | '7d'>('7d');
  const [chartType, setChartType] = useState<'area' | 'candle'>('area');
  const [bars, setBars]           = useState<BarData[]>([]);
  const [predictions, setPredictions] = useState<HorizonPredictions>({ '1d': {}, '3d': {}, '7d': {} });
  const [loading, setLoading]     = useState(true);
  const [range, setRange]         = useState<'1mo' | '3mo'>('1mo');

  useEffect(() => {
    setLoading(true);
    setPredictions({ '1d': {}, '3d': {}, '7d': {} });

    const histFetch = fetchStockHistory(ticker, range, market);

    let predFetch: Promise<HorizonPredictions>;

    if (market === 'VN') {
      // task2_1  → 1-step ahead (day_1)
      // task2_3_k3 → 3 consecutive days (day_1, day_2, day_3)
      // task2_3_k7 → 7 consecutive days (day_1 … day_7)
      predFetch = Promise.all([
        fetchPrediction(ticker, 'task2_1').catch(() => null),
        fetchPrediction(ticker, 'task2_3_k3').catch(() => null),
        fetchPrediction(ticker, 'task2_3_k7').catch(() => null),
      ]).then(([r1, r3k, r7k]) => ({
        '1d': (r1?.predictions  ?? {}) as Record<string, number>,
        '3d': (r3k?.predictions ?? {}) as Record<string, number>,
        '7d': (r7k?.predictions ?? {}) as Record<string, number>,
      }));
    } else {
      // task1_1      → next-day price (day_1)
      // task1_3 k=3  → 3 consecutive days (day_1, day_2, day_3)
      // task1_3 k=7  → 7 consecutive days (day_1 … day_7)
      predFetch = Promise.all([
        fetchNasdaqPrediction(ticker, 1).catch(() => null),
        fetchNasdaqConsecutivePrediction(ticker, 3).catch(() => null),
        fetchNasdaqConsecutivePrediction(ticker, 7).catch(() => null),
      ]).then(([r1, r3k, r7k]) => ({
        '1d': (r1?.predictions  ?? {}) as Record<string, number>,
        '3d': (r3k?.predictions ?? {}) as Record<string, number>,
        '7d': (r7k?.predictions ?? {}) as Record<string, number>,
      }));
    }

    Promise.all([histFetch, predFetch])
      .then(([histBars, preds]) => {
        setBars(histBars as BarData[]);
        setPredictions(preds);
      })
      .catch(console.error)
      .finally(() => setLoading(false));
  }, [ticker, market, range]);

  const chartData = useMemo(
    () => buildChartData(bars, predictions[horizon], horizon),
    [bars, predictions, horizon],
  );

  const lastClose    = bars[bars.length - 1]?.close ?? currentPrice;
  const nDays        = HORIZON_DAYS[horizon];
  const predPrice    = predictions[horizon]?.[`day_${nDays}`];
  const priceDiff    = predPrice && lastClose ? predPrice - lastClose : 0;
  const pricePercent = lastClose ? (priceDiff / lastClose) * 100 : 0;
  const prevClose    = bars[bars.length - 2]?.close;
  const dayChange    = lastClose && prevClose ? ((lastClose - prevClose) / prevClose) * 100 : 0;

  const fmtPrice = (v: number) =>
    market === 'VN'
      ? v.toLocaleString('vi-VN')
      : v.toLocaleString(undefined, { minimumFractionDigits: 2 });

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
            sub:   market === 'VN' ? 'VND' : 'USD',
            color: 'text-text-primary',
          },
          {
            label: 'Day Change',
            value: prevClose ? `${dayChange >= 0 ? '+' : ''}${dayChange.toFixed(2)}%` : '—',
            sub:   'vs yesterday',
            color: dayChange >= 0 ? 'text-pos' : 'text-neg',
          },
          {
            label: `Forecast — ${tradingDateLabel(nDays)}`,
            value: predPrice ? fmtPrice(predPrice) : '—',
            sub:   market === 'VN'
              ? (horizon === '1d' ? 'CNN-LSTM • task2_1' : `CNN-LSTM • task2_3 k=${nDays}`)
              : (horizon === '1d' ? 'LSTM • task1_1'     : `LSTM • task1_3 k=${nDays}`),
            color: 'text-accent-theme',
          },
          {
            label: 'Expected Return',
            value: predPrice ? `${pricePercent >= 0 ? '+' : ''}${pricePercent.toFixed(2)}%` : '—',
            sub:   `by ${tradingDateLabel(nDays)}`,
            color: priceDiff >= 0 ? 'text-pos' : 'text-neg',
          },
        ].map((m) => (
          <div key={m.label} className="theme-card p-5">
            <p className="theme-label mb-1">{m.label}</p>
            <p className={cn("text-xl font-mono font-bold tabular-nums tracking-tighter", m.color)}>
              {m.value}
            </p>
            <p className="text-[10px] text-text-muted font-bold mt-1 uppercase tracking-wider">
              {m.sub}
            </p>
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
                    chartType === ct
                      ? "bg-white text-accent-theme shadow-sm border border-border-theme"
                      : "text-text-muted hover:text-text-secondary",
                  )}>
                  {ct === 'area' ? 'Area' : 'Candle'}
                </button>
              ))}
            </div>
            {/* Horizon */}
            <div className="bg-bg-deep p-1 rounded-xl flex gap-1 border border-border-theme/60">
              {(['1d', '3d', '7d'] as const).map((h) => (
                <button key={h} onClick={() => setHorizon(h)}
                  className={cn(
                    "px-4 py-1.5 text-[10px] font-black uppercase tracking-[0.15em] rounded-lg transition-all",
                    horizon === h
                      ? "bg-white text-accent-theme shadow-sm border border-border-theme"
                      : "text-text-muted hover:text-text-secondary",
                  )}>
                  {h}
                </button>
              ))}
            </div>
            {/* Range */}
            <div className="flex gap-1">
              {(['1mo', '3mo'] as const).map((r) => (
                <button key={r} onClick={() => setRange(r)}
                  className={cn(
                    "px-3 py-1.5 text-[10px] font-bold uppercase tracking-widest rounded-lg border transition-all",
                    range === r
                      ? "bg-white text-accent-theme border-border-theme shadow-sm"
                      : "border-transparent text-text-muted hover:text-text-secondary",
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
                const lastIso = bars[bars.length - 1]?.date_iso;
                if (!lastIso) return dots;
                const horizonPreds = predictions[horizon];
                for (let d = 1; d <= nDays; d++) {
                  const price = horizonPreds?.[`day_${d}`];
                  if (!price) continue;
                  const dt = new Date(lastIso);
                  let count = 0;
                  while (count < d) {
                    dt.setDate(dt.getDate() + 1);
                    if (dt.getDay() !== 0 && dt.getDay() !== 6) count++;
                  }
                  dots.push({ date: dt.toISOString().slice(0, 10), price, label: `+${d}d` });
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
                  tickFormatter={(v) =>
                    market === 'VN' ? (v / 1000).toFixed(0) + 'k' : v.toFixed(0)
                  }
                />
                <Tooltip
                  content={<CustomTooltip market={market} />}
                  cursor={{ stroke: '#4F46E5', strokeWidth: 1, strokeDasharray: '4 4' }}
                />

                {/* Historical price area */}
                <Area
                  type="monotone" dataKey="price"
                  stroke="#4F46E5" strokeWidth={2}
                  fill="url(#gradHist)"
                  dot={false} activeDot={{ r: 4, fill: '#4F46E5', strokeWidth: 0 }}
                  connectNulls={false}
                />

                {/* Forecast area — actual model predictions per day */}
                <Area
                  type="monotone" dataKey="forecast"
                  stroke="#10B981" strokeWidth={2.5} strokeDasharray="6 4"
                  fill="url(#gradFore)"
                  dot={false} activeDot={{ r: 5, fill: '#10B981', strokeWidth: 0 }}
                  connectNulls={false}
                />

                {/* Forecast target reference line */}
                {predPrice && (
                  <ReferenceLine
                    y={predPrice}
                    stroke="#10B981" strokeDasharray="4 4" strokeOpacity={0.5}
                    label={{
                      value: `▶ ${fmtPrice(predPrice)}`,
                      position: 'right',
                      fill: '#10B981', fontSize: 10, fontWeight: 800,
                    }}
                  />
                )}
              </AreaChart>
            </ResponsiveContainer>
          )}
        </div>

        {/* Prediction summary banner (VN only) */}
        {predPrice && market === 'VN' && (
          <AnimatePresence mode="wait">
            <motion.div
              key={horizon}
              initial={{ opacity: 0, y: 6 }}
              animate={{ opacity: 1, y: 0 }}
              transition={{ duration: 0.3 }}
              className={cn(
                "mt-6 flex items-center justify-between px-6 py-4 rounded-2xl border",
                priceDiff >= 0 ? "bg-pos/5 border-pos/20" : "bg-neg/5 border-neg/20",
              )}
            >
              <div className="flex items-center gap-3">
                {priceDiff >= 0
                  ? <TrendingUp size={20} className="text-pos" />
                  : <TrendingDown size={20} className="text-neg" />}
                <div>
                  <p className="text-xs font-black uppercase tracking-widest text-text-muted">
                    CNN-LSTM Forecast
                  </p>
                  <p className={cn("text-sm font-bold", priceDiff >= 0 ? "text-pos" : "text-neg")}>
                    {priceDiff >= 0 ? 'Upward' : 'Downward'} target by{' '}
                    <span className="font-mono">{tradingDateLabel(nDays)}</span>
                    {' — '}
                    <span className="font-mono">{fmtPrice(predPrice)}</span>
                    {' '}({pricePercent >= 0 ? '+' : ''}{pricePercent.toFixed(2)}%)
                  </p>
                </div>
              </div>
              <div className="text-right">
                <p className="text-[9px] text-text-muted font-bold uppercase tracking-wider">
                  from last close
                </p>
                <p className="font-mono text-sm font-bold text-text-primary">{fmtPrice(lastClose)}</p>
              </div>
            </motion.div>
          </AnimatePresence>
        )}
      </div>

    </div>
  );
};
