import React, { useState, useEffect, useMemo } from 'react';
import {
  ResponsiveContainer, AreaChart, Area, BarChart, Bar, Cell,
  XAxis, YAxis, Tooltip, CartesianGrid, ReferenceLine, ReferenceArea, ReferenceDot,
} from 'recharts';
import {
  fetchSignal, fetchStockHistory, fetchRSIData,
  fetchPrediction, fetchNasdaqPrediction,
  SignalData, BarData, RSIData,
} from '../services/marketService';
import { motion, AnimatePresence } from 'motion/react';
import {
  TrendingUp, TrendingDown, Minus, Loader2,
  ShieldCheck, AlertTriangle, Activity, BarChart2,
} from 'lucide-react';
import { cn } from '../lib/utils';

// ── Types ──────────────────────────────────────────────────────────────────────

interface TradingSignalsProps {
  ticker: string;
  market: 'VN' | 'NASDAQ';
}

interface Predictions {
  day1: number | null;
  day3: number | null;
  day7: number | null;
}

type Strength = 'strong' | 'moderate' | 'approaching' | 'none';

// ── Helpers ───────────────────────────────────────────────────────────────────

const WEEKDAYS = ['Sun', 'Mon', 'Tue', 'Wed', 'Thu', 'Fri', 'Sat'];
const MONTHS   = ['Jan', 'Feb', 'Mar', 'Apr', 'May', 'Jun',
                  'Jul', 'Aug', 'Sep', 'Oct', 'Nov', 'Dec'];

function addTradingDays(n: number): string {
  const d = new Date();
  let count = 0;
  while (count < n) {
    d.setDate(d.getDate() + 1);
    const dow = d.getDay();
    if (dow !== 0 && dow !== 6) count++;
  }
  return `${WEEKDAYS[d.getDay()]} ${d.getDate()} ${MONTHS[d.getMonth()]}`;
}

function signalStrength(prob: number, thr: number): Strength {
  if (prob >= thr * 1.15) return 'strong';
  if (prob >= thr)        return 'moderate';
  if (prob >= thr * 0.95) return 'approaching';
  return 'none';
}

function stopLoss(bars: BarData[], lastClose: number): number {
  const recent = bars.slice(-5);
  const atr = recent.reduce((s, b) => s + ((b.high ?? 0) - (b.low ?? 0)), 0) / Math.max(recent.length, 1);
  return lastClose - 3 * atr;
}

const fmtVN  = (v: number) => v.toLocaleString('vi-VN');
const fmtUSD = (v: number) => v.toLocaleString(undefined, { minimumFractionDigits: 2, maximumFractionDigits: 2 });

// ── Signal palette ─────────────────────────────────────────────────────────────

const SIGNAL_STYLE = {
  BUY:  { bg: 'bg-pos/10',          border: 'border-pos/20',          text: 'text-pos',         fill: '#10B981' },
  SELL: { bg: 'bg-neg/10',          border: 'border-neg/20',          text: 'text-neg',          fill: '#EF4444' },
  HOLD: { bg: 'bg-amber-500/10',    border: 'border-amber-500/20',    text: 'text-amber-500',    fill: '#F59E0B' },
} as const;

const STRENGTH_LABEL: Record<Strength, string> = {
  strong:      'Strong',
  moderate:    'Moderate',
  approaching: 'Approaching',
  none:        '—',
};

// ── Sub-components ─────────────────────────────────────────────────────────────

const ProbBar: React.FC<{ label: string; prob: number; threshold: number; color: string }> = ({
  label, prob, threshold, color,
}) => {
  const pct    = Math.round(prob * 100);
  const thrPct = Math.round(threshold * 100);
  const active     = prob >= threshold;
  const approaching = !active && prob >= threshold * 0.95;

  return (
    <div className="space-y-1.5">
      <div className="flex items-center justify-between text-[10px] font-bold uppercase tracking-widest text-text-muted">
        <span>{label}</span>
        <span className={cn('font-mono text-xs', active ? '' : approaching ? 'text-amber-500' : '')}
              style={active ? { color } : undefined}>
          {(prob * 100).toFixed(1)}%
          <span className="opacity-50 ml-1">/ thr {thrPct}%</span>
        </span>
      </div>
      <div className="relative h-3 bg-bg-deep rounded-full border border-border-theme/50">
        <div
          className="h-full rounded-full transition-all duration-700"
          style={{
            width:      `${pct}%`,
            background: active ? color : approaching ? '#F59E0B' : '#94A3B8',
            opacity:    active ? 1 : 0.5,
          }}
        />
        {/* Threshold tick */}
        <div
          className="absolute top-[-3px] bottom-[-3px] w-0.5 bg-text-primary/60 rounded-full"
          style={{ left: `${thrPct}%` }}
        />
      </div>
    </div>
  );
};

const ChartTooltip = ({ active, payload, market }: any) => {
  if (!active || !payload?.length) return null;
  const d = payload[0].payload;
  const val = d.forecast ?? d.price;
  if (val == null) return null;
  const fmt = market === 'VN' ? fmtVN : fmtUSD;
  return (
    <div className="bg-bg-surface border border-border-theme px-4 py-3 rounded-xl shadow-xl text-xs font-mono">
      <p className="text-text-muted mb-1 font-sans font-bold uppercase tracking-wider">{d.date}</p>
      <p className="text-text-primary font-bold text-sm">{fmt(val)}</p>
      {d.isForecastDot && (
        <p className="text-pos text-[10px] font-sans font-bold mt-1 uppercase tracking-wider">{d.horizonLabel}</p>
      )}
    </div>
  );
};

const RSITooltip = ({ active, payload }: any) => {
  if (!active || !payload?.length) return null;
  const d = payload[0].payload as RSIData;
  return (
    <div className="bg-bg-surface border border-border-theme px-3 py-2 rounded-xl shadow-xl text-xs">
      <p className="text-text-muted font-bold uppercase tracking-wider mb-1">{d.date}</p>
      {d.rsi  != null && <p className="font-mono text-accent-theme">RSI <span className="font-bold">{d.rsi.toFixed(1)}</span></p>}
      {d.ma50 != null && <p className="font-mono text-text-secondary">MA50 <span className="font-bold">{d.ma50.toFixed(0)}</span></p>}
    </div>
  );
};

const VolumeTooltip = ({ active, payload }: any) => {
  if (!active || !payload?.length) return null;
  const d = payload[0].payload as BarData;
  const vol = d.volume ?? 0;
  return (
    <div className="bg-bg-surface border border-border-theme px-3 py-2 rounded-xl shadow-xl text-xs">
      <p className="text-text-muted font-bold uppercase tracking-wider mb-1">{d.date}</p>
      <p className="font-mono text-text-primary font-bold">
        {vol >= 1_000_000
          ? (vol / 1_000_000).toFixed(2) + 'M'
          : (vol / 1000).toFixed(0) + 'K'}
      </p>
    </div>
  );
};

// ── Main component ─────────────────────────────────────────────────────────────

export const TradingSignals: React.FC<TradingSignalsProps> = ({ ticker, market }) => {
  const [signal,  setSignal]  = useState<SignalData | null>(null);
  const [bars,    setBars]    = useState<BarData[]>([]);
  const [rsiData, setRsiData] = useState<RSIData[]>([]);
  const [preds,   setPreds]   = useState<Predictions>({ day1: null, day3: null, day7: null });
  const [range,   setRange]   = useState<'1mo' | '3mo'>('1mo');
  const [loading, setLoading] = useState(true);

  useEffect(() => {
    setLoading(true);
    setSignal(null);
    setPreds({ day1: null, day3: null, day7: null });

    const histFetch = fetchStockHistory(ticker, range, market);
    const rsiFetch  = fetchRSIData(ticker, market);

    let predFetch: Promise<Predictions>;
    let sigFetch: Promise<SignalData | null>;

    if (market === 'VN') {
      sigFetch  = fetchSignal(ticker).catch(() => null);
      predFetch = Promise.all([
        fetchPrediction(ticker, 'task2_1').catch(() => null),
        fetchPrediction(ticker, 'task2_2_k3').catch(() => null),
        fetchPrediction(ticker, 'task2_2_k7').catch(() => null),
      ]).then(([r1, r3, r7]) => {
        const first = (r: any): number | null =>
          r ? (Object.values(r.predictions as Record<string, number>)[0] ?? null) : null;
        return { day1: first(r1), day3: first(r3), day7: first(r7) };
      });
    } else {
      sigFetch  = Promise.resolve(null);
      predFetch = fetchNasdaqPrediction(ticker)
        .then(r => ({ day1: r.predictions.day_1 ?? null, day3: null, day7: null }))
        .catch(() => ({ day1: null, day3: null, day7: null }));
    }

    Promise.all([histFetch, rsiFetch, predFetch, sigFetch])
      .then(([h, rsi, p, s]) => {
        setBars(h as BarData[]);
        setRsiData(rsi as RSIData[]);
        setPreds(p);
        setSignal(s);
      })
      .catch(console.error)
      .finally(() => setLoading(false));
  }, [ticker, market, range]);

  // ── Derived values ──────────────────────────────────────────────────────────

  const lastClose  = bars[bars.length - 1]?.close ?? 0;
  const rsiFiltered = rsiData.filter(d => d.rsi != null);
  const currentRSI  = rsiFiltered[rsiFiltered.length - 1]?.rsi ?? null;
  const sl         = lastClose > 0 ? stopLoss(bars, lastClose) : null;
  const fmt        = market === 'VN' ? fmtVN : fmtUSD;
  const unit       = market === 'VN' ? 'VND' : 'USD';

  const ret = (pred: number | null): number | null =>
    pred != null && lastClose > 0 ? ((pred - lastClose) / lastClose) * 100 : null;

  const sig     = signal?.signal ?? 'HOLD';
  const style   = SIGNAL_STYLE[sig];
  const buyStr  = signal ? signalStrength(signal.buy_prob,  signal.threshold) : 'none';
  const sellStr = signal ? signalStrength(signal.sell_prob, signal.threshold) : 'none';

  // ── Chart data ──────────────────────────────────────────────────────────────

  const chartData = useMemo(() => {
    if (!bars.length) return [];

    const hist = bars.slice(-40).map(b => ({
      date:          b.date,
      price:         b.close ?? null,
      forecast:      null as number | null,
      isForecastDot: false,
      horizonLabel:  '',
    }));

    // Bridge: last historical bar starts the forecast series
    const bridgeIdx  = hist.length - 1;
    hist[bridgeIdx]  = { ...hist[bridgeIdx], forecast: hist[bridgeIdx].price };

    // Discrete forecast dots (no interpolation between them)
    const dots = [
      { n: 1, val: preds.day1, label: '+1 Day'  },
      { n: 3, val: preds.day3, label: '+3 Days' },
      { n: 7, val: preds.day7, label: '+7 Days' },
    ]
      .filter(d => d.val != null)
      .map(d => ({
        date:          addTradingDays(d.n),
        price:         null as number | null,
        forecast:      d.val as number,
        isForecastDot: true,
        horizonLabel:  d.label,
      }));

    return [...hist, ...dots];
  }, [bars, preds]);

  const yDomain = useMemo(() => {
    const vals = chartData.flatMap(d => [d.price, d.forecast]).filter(v => v != null) as number[];
    if (!vals.length) return ['auto' as const, 'auto' as const];
    const min = Math.min(...vals);
    const max = Math.max(...vals);
    const pad = (max - min) * 0.1;
    return [min - pad, max + pad] as [number, number];
  }, [chartData]);

  const vol20avg = useMemo(() => {
    const vols = bars.slice(-20).map(b => b.volume ?? 0);
    return vols.length ? vols.reduce((a, b) => a + b, 0) / vols.length : 0;
  }, [bars]);

  // ── Loading ─────────────────────────────────────────────────────────────────

  if (loading) {
    return (
      <div className="flex flex-col items-center justify-center h-[600px] text-text-secondary">
        <Loader2 className="animate-spin mb-4 text-accent-theme" size={40} />
        <p className="font-mono text-xs uppercase tracking-widest">Loading Signal Data…</p>
      </div>
    );
  }

  // ── Render ──────────────────────────────────────────────────────────────────

  return (
    <div className="space-y-6 pb-20">

      {/* ════════════════════════════════════════════════════════════════════
          SECTION 1 — Signal Command
      ════════════════════════════════════════════════════════════════════ */}
      <AnimatePresence mode="wait">
        <motion.div
          key={`${ticker}-${market}-${sig}`}
          initial={{ opacity: 0, y: 8 }}
          animate={{ opacity: 1, y: 0 }}
          transition={{ duration: 0.3 }}
          className="grid grid-cols-5 gap-4"
        >

          {/* Col 1+2: Primary signal */}
          {market === 'VN' && signal ? (
            <div className={cn('col-span-2 theme-card p-6 flex flex-col gap-4 border', style.bg, style.border)}>
              <div className="flex items-start justify-between">
                <div>
                  <p className="theme-label mb-2">Active Signal</p>
                  <div className="flex items-center gap-3">
                    {sig === 'BUY'  && <TrendingUp  size={32} className="text-pos" />}
                    {sig === 'SELL' && <TrendingDown size={32} className="text-neg" />}
                    {sig === 'HOLD' && <Minus        size={32} className="text-amber-500" />}
                    <div>
                      <p className={cn('text-4xl font-black tracking-tighter leading-none', style.text)}>{sig}</p>
                      <p className={cn('text-[11px] font-bold uppercase tracking-widest mt-1', style.text, 'opacity-70')}>
                        {sig === 'BUY'
                          ? STRENGTH_LABEL[buyStr]  + ' Signal'
                          : sig === 'SELL'
                          ? STRENGTH_LABEL[sellStr] + ' Signal'
                          : 'No clear direction'}
                      </p>
                    </div>
                  </div>
                </div>
                <div className="flex flex-col items-end gap-1">
                  <div className="flex items-center gap-1.5 px-2 py-1 bg-bg-surface rounded-lg border border-border-theme">
                    <ShieldCheck size={11} className="text-accent-theme" />
                    <span className="text-[9px] font-black text-accent-theme uppercase tracking-widest">Task 3 CNN-LSTM</span>
                  </div>
                  <p className="text-[9px] text-text-muted font-bold uppercase tracking-widest">30-day lookback · 24 features</p>
                </div>
              </div>

              {/* Model performance grid */}
              <div className="grid grid-cols-2 gap-2 pt-3 border-t border-border-theme/40">
                {[
                  { label: 'BUY AUC',   val: signal.buy_auc.toFixed(3),   color: 'text-pos' },
                  { label: 'SELL AUC',  val: signal.sell_auc.toFixed(3),  color: 'text-neg' },
                  { label: 'BUY F1',    val: signal.buy_f1.toFixed(3),    color: 'text-pos' },
                  { label: 'SELL F1',   val: signal.sell_f1.toFixed(3),   color: 'text-neg' },
                ].map(m => (
                  <div key={m.label} className="flex items-center justify-between px-3 py-2 rounded-lg bg-bg-deep border border-border-theme/40">
                    <span className="text-[9px] font-black uppercase tracking-widest text-text-muted">{m.label}</span>
                    <span className={cn('font-mono text-xs font-bold', m.color)}>{m.val}</span>
                  </div>
                ))}
              </div>
            </div>
          ) : (
            <div className="col-span-2 theme-card p-6 flex flex-col items-center justify-center gap-3 border border-border-theme/50">
              <AlertTriangle size={28} className="text-amber-500" />
              <p className="text-sm font-bold text-text-secondary text-center">
                Task 3 signal models are trained on VN market only.
              </p>
              <p className="text-[10px] text-text-muted font-bold uppercase tracking-widest text-center">
                Price forecast + technical analysis below
              </p>
            </div>
          )}

          {/* Col 3+4: Probability meters / NASDAQ forecast */}
          {market === 'VN' && signal ? (
            <div className="col-span-2 theme-card p-6 flex flex-col justify-center gap-5">
              <p className="theme-label">Signal Probability Meters</p>
              <div className="space-y-5">
                <ProbBar label="BUY probability"  prob={signal.buy_prob}  threshold={signal.threshold} color="#10B981" />
                <ProbBar label="SELL probability" prob={signal.sell_prob} threshold={signal.threshold} color="#EF4444" />
              </div>
              <p className="text-[9px] text-text-muted font-medium leading-relaxed border-t border-border-theme/40 pt-3">
                Probability of local price <span className="text-pos font-bold">minimum</span> (BUY) /{' '}
                <span className="text-neg font-bold">maximum</span> (SELL) within a 5-day window.
                Vertical tick = per-ticker calibrated threshold.
              </p>
            </div>
          ) : (
            <div className="col-span-2 theme-card p-6 flex flex-col justify-center gap-4">
              <p className="theme-label">Task 1 LSTM Forecast</p>
              {preds.day1 != null ? (
                <div className="space-y-3">
                  <div className="flex items-baseline justify-between">
                    <span className="text-[10px] font-black uppercase tracking-widest text-text-muted">+1 Day Target</span>
                    <span className="font-mono text-xl font-bold text-accent-theme">{fmtUSD(preds.day1)}</span>
                  </div>
                  <div className={cn(
                    'px-3 py-2 rounded-lg text-sm font-bold flex items-center justify-between',
                    (ret(preds.day1) ?? 0) >= 0 ? 'bg-pos/10 text-pos' : 'bg-neg/10 text-neg'
                  )}>
                    <span className="text-[10px] uppercase tracking-widest font-black opacity-70">Expected Return</span>
                    <span>{(ret(preds.day1) ?? 0) >= 0 ? '+' : ''}{(ret(preds.day1) ?? 0).toFixed(2)}%</span>
                  </div>
                  <p className="text-[9px] text-text-muted font-bold uppercase tracking-widest">{addTradingDays(1)}</p>
                </div>
              ) : (
                <p className="text-text-muted text-sm">No prediction available</p>
              )}
            </div>
          )}

          {/* Col 5: Forecast target mini-cards */}
          <div className="col-span-1 theme-card p-5 flex flex-col gap-3">
            <p className="theme-label">Targets</p>
            {market === 'VN' ? (
              [
                { n: 1, val: preds.day1, label: '+1 Day'  },
                { n: 3, val: preds.day3, label: '+3 Days' },
                { n: 7, val: preds.day7, label: '+7 Days' },
              ].map(({ n, val, label }) => {
                const r  = ret(val);
                const up = (r ?? 0) >= 0;
                return (
                  <div key={n} className={cn(
                    'rounded-xl p-3 border',
                    val != null
                      ? up ? 'bg-pos/5 border-pos/15' : 'bg-neg/5 border-neg/15'
                      : 'bg-bg-deep border-border-theme/40'
                  )}>
                    <p className="text-[9px] font-black uppercase tracking-widest text-text-muted mb-1">{label}</p>
                    {val != null ? (
                      <>
                        <p className={cn('font-mono text-xs font-bold', up ? 'text-pos' : 'text-neg')}>{fmt(val)}</p>
                        <p className={cn('text-[10px] font-bold', up ? 'text-pos' : 'text-neg')}>
                          {up ? '+' : ''}{r!.toFixed(2)}%
                        </p>
                      </>
                    ) : (
                      <p className="text-text-muted text-xs font-mono">—</p>
                    )}
                  </div>
                );
              })
            ) : (
              <>
                {[{ n: 1, val: preds.day1, label: '+1 Day' }].map(({ n, val, label }) => {
                  const r  = ret(val);
                  const up = (r ?? 0) >= 0;
                  return (
                    <div key={n} className={cn(
                      'rounded-xl p-3 border',
                      val != null
                        ? up ? 'bg-pos/5 border-pos/15' : 'bg-neg/5 border-neg/15'
                        : 'bg-bg-deep border-border-theme/40'
                    )}>
                      <p className="text-[9px] font-black uppercase tracking-widest text-text-muted mb-1">{label}</p>
                      {val != null ? (
                        <>
                          <p className={cn('font-mono text-xs font-bold', up ? 'text-pos' : 'text-neg')}>{fmtUSD(val)}</p>
                          <p className={cn('text-[10px] font-bold', up ? 'text-pos' : 'text-neg')}>
                            {up ? '+' : ''}{r!.toFixed(2)}%
                          </p>
                        </>
                      ) : (
                        <p className="text-text-muted text-xs font-mono">—</p>
                      )}
                    </div>
                  );
                })}
                <p className="text-[9px] text-text-muted text-center font-bold uppercase tracking-widest mt-auto">
                  Task 1 models<br />next day only
                </p>
              </>
            )}
          </div>

        </motion.div>
      </AnimatePresence>

      {/* ════════════════════════════════════════════════════════════════════
          SECTION 2 — Price + Signal + Forecast Chart
      ════════════════════════════════════════════════════════════════════ */}
      <div className="theme-card p-8">
        <div className="flex items-center justify-between mb-6">
          <div>
            <h3 className="theme-label mb-1">Price Trajectory + Signal + Forecast Targets</h3>
            <div className="flex items-center gap-4 text-xs text-text-muted font-medium">
              <span className="flex items-center gap-1.5">
                <span className="w-8 h-0.5 bg-accent-theme inline-block rounded" />
                Historical
              </span>
              {market === 'VN' && sig === 'BUY'  && <span className="flex items-center gap-1 text-pos font-bold">▲ BUY today</span>}
              {market === 'VN' && sig === 'SELL' && <span className="flex items-center gap-1 text-neg font-bold">▼ SELL today</span>}
              {market === 'VN' && sig === 'HOLD' && <span className="flex items-center gap-1 text-amber-500 font-bold">● HOLD today</span>}
              <span className="flex items-center gap-1.5">
                <span className="w-2.5 h-2.5 rounded-full bg-pos inline-block border-2 border-pos/30" />
                Forecast targets
              </span>
            </div>
          </div>
          <div className="flex gap-1">
            {(['1mo', '3mo'] as const).map(r => (
              <button key={r} onClick={() => setRange(r)}
                className={cn(
                  'px-3 py-1.5 text-[10px] font-bold uppercase tracking-widest rounded-lg border transition-all',
                  range === r
                    ? 'bg-white text-accent-theme border-border-theme shadow-sm'
                    : 'border-transparent text-text-muted hover:text-text-secondary'
                )}>
                {r.replace('mo', 'M')}
              </button>
            ))}
          </div>
        </div>

        <div className="h-[340px]">
          <ResponsiveContainer width="100%" height="100%">
            <AreaChart data={chartData} margin={{ top: 10, right: 70, bottom: 0, left: 0 }}>
              <defs>
                <linearGradient id="gradHist" x1="0" y1="0" x2="0" y2="1">
                  <stop offset="0%"   stopColor="#4F46E5" stopOpacity={0.15} />
                  <stop offset="100%" stopColor="#4F46E5" stopOpacity={0} />
                </linearGradient>
              </defs>
              <CartesianGrid strokeDasharray="3 3" stroke="#F1F5F9" vertical={false} />
              <XAxis
                dataKey="date" axisLine={false} tickLine={false}
                tick={{ fill: '#94A3B8', fontSize: 10, fontWeight: 700 }}
                minTickGap={40} dy={10}
              />
              <YAxis
                domain={yDomain} axisLine={false} tickLine={false}
                tick={{ fill: '#94A3B8', fontSize: 10, fontWeight: 700 }}
                orientation="right" dx={12}
                tickFormatter={v => market === 'VN' ? (v / 1000).toFixed(0) + 'k' : v.toFixed(0)}
              />
              <Tooltip content={<ChartTooltip market={market} />}
                cursor={{ stroke: '#4F46E5', strokeWidth: 1, strokeDasharray: '4 4' }} />

              {/* Historical area */}
              <Area
                type="monotone" dataKey="price"
                stroke="#4F46E5" strokeWidth={2}
                fill="url(#gradHist)"
                dot={false}
                activeDot={{ r: 4, fill: '#4F46E5', strokeWidth: 0 }}
                connectNulls={false}
              />

              {/* Forecast dots — rendered via custom dot, no line between them */}
              <Area
                type="monotone" dataKey="forecast"
                stroke="#10B981" strokeWidth={0}
                fill="none"
                dot={(props: any) => {
                  if (!props.payload?.isForecastDot) return <g key={props.key} />;
                  return (
                    <circle
                      key={props.key}
                      cx={props.cx} cy={props.cy} r={7}
                      fill="#10B981" stroke="#fff" strokeWidth={2.5}
                    />
                  );
                }}
                activeDot={{ r: 8, fill: '#10B981', stroke: '#fff', strokeWidth: 2 }}
                connectNulls={false}
              />

              {/* Signal annotation at today's close */}
              {market === 'VN' && signal && lastClose > 0 && (() => {
                const bridgeDate = chartData.find(d => d.price != null && d.forecast != null)?.date ?? '';
                const label = sig === 'BUY' ? '▲' : sig === 'SELL' ? '▼' : '●';
                return (
                  <ReferenceDot
                    x={bridgeDate} y={lastClose}
                    r={9} fill={style.fill} stroke="#fff" strokeWidth={2}
                    label={{ value: label, position: 'top', fill: style.fill, fontSize: 13, fontWeight: 900 }}
                  />
                );
              })()}

              {/* Dashed reference lines at each forecast level */}
              {[
                { val: preds.day1, label: '+1d' },
                { val: preds.day3, label: '+3d' },
                { val: preds.day7, label: '+7d' },
              ].filter(d => d.val != null).map(({ val, label }) => (
                <ReferenceLine
                  key={label} y={val!}
                  stroke="#10B981" strokeDasharray="4 3" strokeOpacity={0.35}
                  label={{
                    value: `${label} ${market === 'VN' ? (val! / 1000).toFixed(0) + 'k' : val!.toFixed(0)}`,
                    position: 'right', fill: '#10B981', fontSize: 9, fontWeight: 800,
                  }}
                />
              ))}
            </AreaChart>
          </ResponsiveContainer>
        </div>
      </div>

      {/* ════════════════════════════════════════════════════════════════════
          SECTION 3 — RSI + Volume
      ════════════════════════════════════════════════════════════════════ */}
      <div className="grid grid-cols-2 gap-4">

        {/* RSI(14) */}
        <div className="theme-card p-6">
          <div className="flex items-center justify-between mb-4">
            <div className="flex items-center gap-2">
              <Activity size={15} className="text-accent-theme" />
              <h3 className="theme-label mb-0">RSI(14) Momentum</h3>
            </div>
            {currentRSI != null && (
              <div className={cn(
                'px-3 py-1 rounded-lg text-xs font-black font-mono border',
                currentRSI > 70 ? 'bg-neg/10 text-neg border-neg/20'
                : currentRSI < 30 ? 'bg-pos/10 text-pos border-pos/20'
                : 'bg-accent-theme/10 text-accent-theme border-accent-theme/20'
              )}>
                {currentRSI.toFixed(1)}
                <span className="text-[9px] ml-1.5 opacity-60 font-bold uppercase tracking-wider">
                  {currentRSI > 70 ? 'Overbought' : currentRSI < 30 ? 'Oversold' : 'Neutral'}
                </span>
              </div>
            )}
          </div>

          <div className="h-[220px]">
            <ResponsiveContainer width="100%" height="100%">
              <AreaChart data={rsiData.slice(-60)} margin={{ top: 5, right: 10, bottom: 0, left: 0 }}>
                <ReferenceArea y1={70} y2={100} fill="#EF4444" fillOpacity={0.07} />
                <ReferenceArea y1={0}  y2={30}  fill="#10B981" fillOpacity={0.07} />
                <defs>
                  <linearGradient id="gradRSI" x1="0" y1="0" x2="0" y2="1">
                    <stop offset="0%"   stopColor="#4F46E5" stopOpacity={0.2} />
                    <stop offset="100%" stopColor="#4F46E5" stopOpacity={0} />
                  </linearGradient>
                </defs>
                <CartesianGrid strokeDasharray="3 3" stroke="#F1F5F9" vertical={false} />
                <XAxis
                  dataKey="date" axisLine={false} tickLine={false}
                  tick={{ fill: '#94A3B8', fontSize: 9, fontWeight: 700 }}
                  minTickGap={30} dy={8}
                />
                <YAxis
                  domain={[0, 100]} axisLine={false} tickLine={false}
                  tick={{ fill: '#94A3B8', fontSize: 9, fontWeight: 700 }}
                  orientation="right" dx={8}
                  ticks={[0, 30, 50, 70, 100]}
                />
                <Tooltip content={<RSITooltip />}
                  cursor={{ stroke: '#4F46E5', strokeWidth: 1, strokeDasharray: '4 4' }} />
                <ReferenceLine y={70} stroke="#EF4444" strokeDasharray="4 3" strokeOpacity={0.6}
                  label={{ value: '70', position: 'right', fill: '#EF4444', fontSize: 9, fontWeight: 800 }} />
                <ReferenceLine y={30} stroke="#10B981" strokeDasharray="4 3" strokeOpacity={0.6}
                  label={{ value: '30', position: 'right', fill: '#10B981', fontSize: 9, fontWeight: 800 }} />
                <Area
                  type="monotone" dataKey="rsi"
                  stroke="#4F46E5" strokeWidth={1.5}
                  fill="url(#gradRSI)"
                  dot={false}
                  activeDot={{ r: 3, fill: '#4F46E5', strokeWidth: 0 }}
                  connectNulls
                />
              </AreaChart>
            </ResponsiveContainer>
          </div>
        </div>

        {/* Volume */}
        <div className="theme-card p-6">
          <div className="flex items-center justify-between mb-4">
            <div className="flex items-center gap-2">
              <BarChart2 size={15} className="text-accent-theme" />
              <h3 className="theme-label mb-0">Volume Analysis</h3>
            </div>
            {vol20avg > 0 && (
              <div className="px-3 py-1 rounded-lg text-xs font-mono font-bold bg-bg-deep border border-border-theme text-text-secondary">
                20d avg&nbsp;
                {vol20avg >= 1_000_000
                  ? (vol20avg / 1_000_000).toFixed(1) + 'M'
                  : (vol20avg / 1000).toFixed(0) + 'K'}
              </div>
            )}
          </div>

          <div className="h-[220px]">
            <ResponsiveContainer width="100%" height="100%">
              <BarChart data={bars.slice(-40)} margin={{ top: 5, right: 10, bottom: 0, left: 0 }}>
                <CartesianGrid strokeDasharray="3 3" stroke="#F1F5F9" vertical={false} />
                <XAxis
                  dataKey="date" axisLine={false} tickLine={false}
                  tick={{ fill: '#94A3B8', fontSize: 9, fontWeight: 700 }}
                  minTickGap={30} dy={8}
                />
                <YAxis
                  axisLine={false} tickLine={false}
                  tick={{ fill: '#94A3B8', fontSize: 9, fontWeight: 700 }}
                  orientation="right" dx={8}
                  tickFormatter={v =>
                    v >= 1_000_000
                      ? (v / 1_000_000).toFixed(0) + 'M'
                      : (v / 1000).toFixed(0) + 'K'
                  }
                />
                <Tooltip content={<VolumeTooltip />} cursor={{ fill: 'rgba(99,102,241,0.05)' }} />
                {vol20avg > 0 && (
                  <ReferenceLine y={vol20avg} stroke="#94A3B8" strokeDasharray="4 3" strokeOpacity={0.6}
                    label={{ value: 'Avg', position: 'right', fill: '#94A3B8', fontSize: 9, fontWeight: 800 }} />
                )}
                <Bar dataKey="volume" radius={[2, 2, 0, 0]}>
                  {bars.slice(-40).map((b, i) => (
                    <Cell
                      key={i}
                      fill={(b.close ?? 0) >= (b.open ?? 0) ? '#10B981' : '#EF4444'}
                      opacity={0.75}
                    />
                  ))}
                </Bar>
              </BarChart>
            </ResponsiveContainer>
          </div>
        </div>
      </div>

      {/* ════════════════════════════════════════════════════════════════════
          SECTION 4 — Trading Plan
      ════════════════════════════════════════════════════════════════════ */}
      {market === 'VN' && signal ? (
        <div className="grid grid-cols-4 gap-4">
          {/* Entry Zone */}
          <div className="theme-card p-5">
            <div className="flex items-center gap-2 mb-3">
              <TrendingUp size={14} className="text-accent-theme" />
              <p className="theme-label mb-0">Entry Zone</p>
            </div>
            {sig === 'BUY' && (
              <>
                <p className="text-sm font-bold text-pos mb-1">Days 1–3, buy at market</p>
                <p className="text-[10px] text-text-muted font-medium leading-relaxed whitespace-pre-line">
                  {`${addTradingDays(1)} →\n${addTradingDays(3)}`}
                </p>
              </>
            )}
            {sig === 'SELL' && (
              <>
                <p className="text-sm font-bold text-neg mb-1">Avoid new entries</p>
                <p className="text-[10px] text-text-muted font-medium leading-relaxed">
                  Selling pressure detected. Wait for reversal signal.
                </p>
              </>
            )}
            {sig === 'HOLD' && (
              <>
                <p className="text-sm font-bold text-amber-500 mb-1">Monitor</p>
                <p className="text-[10px] text-text-muted font-medium leading-relaxed">
                  No clear edge detected. Wait for signal confirmation.
                </p>
              </>
            )}
          </div>

          {/* Price Targets */}
          <div className="theme-card p-5">
            <div className="flex items-center gap-2 mb-3">
              <Activity size={14} className="text-accent-theme" />
              <p className="theme-label mb-0">Price Targets</p>
            </div>
            <div className="space-y-1.5">
              {[
                { label: '+1d', val: preds.day1 },
                { label: '+3d', val: preds.day3 },
                { label: '+7d', val: preds.day7 },
              ].map(({ label, val }) => {
                const r  = ret(val);
                const up = (r ?? 0) >= 0;
                return (
                  <div key={label} className="flex items-center justify-between text-xs">
                    <span className="text-[9px] font-black uppercase tracking-widest text-text-muted w-8">{label}</span>
                    {val != null ? (
                      <span className={cn('font-mono font-bold', up ? 'text-pos' : 'text-neg')}>
                        {fmt(val)}
                        <span className="text-[9px] ml-1 opacity-70">({up ? '+' : ''}{r!.toFixed(1)}%)</span>
                      </span>
                    ) : (
                      <span className="text-text-muted font-mono">—</span>
                    )}
                  </div>
                );
              })}
            </div>
            <p className="text-[9px] text-text-muted font-bold uppercase tracking-widest mt-2 border-t border-border-theme/40 pt-2">
              {unit} · CNN-LSTM
            </p>
          </div>

          {/* Stop Loss */}
          <div className="theme-card p-5">
            <div className="flex items-center gap-2 mb-3">
              <ShieldCheck size={14} className="text-accent-theme" />
              <p className="theme-label mb-0">Stop Loss</p>
            </div>
            {sig === 'BUY' && sl != null ? (
              <>
                <p className="text-sm font-bold text-neg mb-1">{fmtVN(Math.round(sl))}</p>
                <p className="text-[10px] text-text-muted font-medium leading-relaxed">
                  Last close − 3× ATR(5)
                  <br />
                  <span className="font-mono">{fmtVN(lastClose)}</span> − 3 × avg range
                </p>
              </>
            ) : (
              <>
                <p className="text-sm font-bold text-text-muted mb-1">—</p>
                <p className="text-[10px] text-text-muted font-medium">
                  Stop loss applies to BUY positions only.
                </p>
              </>
            )}
          </div>

          {/* Exit Trigger */}
          <div className="theme-card p-5">
            <div className="flex items-center gap-2 mb-3">
              <AlertTriangle size={14} className="text-accent-theme" />
              <p className="theme-label mb-0">Exit Trigger</p>
            </div>
            {sig === 'BUY' ? (
              <>
                <p className="text-sm font-bold text-pos mb-1">RSI &gt; 70 or target hit</p>
                <p className="text-[10px] text-text-muted font-medium leading-relaxed">
                  RSI overbought{currentRSI != null ? ` (now ${currentRSI.toFixed(1)})` : ''}<br />
                  OR price ≥ {preds.day7 != null ? fmt(preds.day7) + ' (+7d)' : 'day 7 target'}
                </p>
              </>
            ) : sig === 'SELL' ? (
              <>
                <p className="text-sm font-bold text-neg mb-1">RSI &lt; 30 or target hit</p>
                <p className="text-[10px] text-text-muted font-medium leading-relaxed">
                  RSI oversold{currentRSI != null ? ` (now ${currentRSI.toFixed(1)})` : ''}<br />
                  OR price ≤ {preds.day7 != null ? fmt(preds.day7) + ' (+7d)' : 'day 7 target'}
                </p>
              </>
            ) : (
              <>
                <p className="text-sm font-bold text-amber-500 mb-1">Re-run in 3 days</p>
                <p className="text-[10px] text-text-muted font-medium">
                  Check signal again on<br />{addTradingDays(3)}
                </p>
              </>
            )}
          </div>
        </div>
      ) : market === 'NASDAQ' ? (
        <div className="theme-card p-5 flex items-center gap-4 border border-border-theme/50">
          <AlertTriangle size={18} className="text-amber-500 shrink-0" />
          <p className="text-sm text-text-secondary font-medium">
            Trading plan cards are derived from Task 3 buy/sell signals, which are trained on VN market tickers only.
            Use the RSI chart and Task 1 forecast above for directional context on NASDAQ stocks.
          </p>
        </div>
      ) : null}

    </div>
  );
};
