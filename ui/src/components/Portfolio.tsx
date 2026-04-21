import React, { useState, useEffect, useMemo } from 'react';
import { PieChart, Pie, Cell, Tooltip, ResponsiveContainer } from 'recharts';
import { fetchPortfolio, PortfolioData } from '../services/marketService';
import { Loader2, TrendingUp, Shield, SlidersHorizontal } from 'lucide-react';
import { cn } from '../lib/utils';

const COLORS = ['#4F46E5', '#10B981', '#F59E0B', '#EF4444', '#8B5CF6', '#06B6D4'];

function lerp(a: number, b: number, t: number) {
  return a + (b - a) * t;
}

export const Portfolio: React.FC = () => {
  const [data, setData]         = useState<{ prudent: PortfolioData; risk_taking: PortfolioData } | null>(null);
  const [loading, setLoading]   = useState(true);
  const [targetPct, setTargetPct] = useState(12); // slider value in %

  useEffect(() => {
    fetchPortfolio()
      .then(setData)
      .catch(console.error)
      .finally(() => setLoading(false));
  }, []);

  // Slider bounds derived from actual portfolio data
  const minReturn = useMemo(() => data ? Math.floor(data.prudent.expected_return * 100 * 10) / 10 : 5,    [data]);
  const maxReturn = useMemo(() => data ? Math.ceil(data.risk_taking.expected_return * 100 * 10) / 10 : 20, [data]);

  // Keep targetPct within bounds when data loads
  useEffect(() => {
    if (data) setTargetPct(Math.round((minReturn + maxReturn) / 2 * 10) / 10);
  }, [minReturn, maxReturn]);

  // Blend factor: 0 = fully prudent, 1 = fully risk-taking
  const blend = useMemo(() => {
    if (!data) return 0;
    const range = maxReturn - minReturn;
    return range > 0 ? Math.max(0, Math.min(1, (targetPct - minReturn) / range)) : 0;
  }, [targetPct, minReturn, maxReturn, data]);

  // Blended portfolio metrics
  const blended = useMemo(() => {
    if (!data) return null;
    const p = data.prudent;
    const r = data.risk_taking;
    return {
      expected_return:   lerp(p.expected_return,   r.expected_return,   blend),
      annual_volatility: lerp(p.annual_volatility, r.annual_volatility, blend),
      sharpe_ratio:      lerp(p.sharpe_ratio,      r.sharpe_ratio,      blend),
    };
  }, [blend, data]);

  // Active portfolio for pie chart
  const activePortfolio: PortfolioData | null = useMemo(() => {
    if (!data) return null;
    return blend < 0.5 ? data.prudent : data.risk_taking;
  }, [blend, data]);

  const recommendation = blend < 0.35 ? 'Prudent' : blend > 0.65 ? 'Risk-Taking' : 'Balanced Mix';
  const recColor       = blend < 0.35 ? 'text-accent-theme' : blend > 0.65 ? 'text-neg' : 'text-pos';

  if (loading) {
    return (
      <div className="flex flex-col items-center justify-center h-[400px] text-text-secondary">
        <Loader2 className="animate-spin mb-4 text-accent-theme" size={40} />
        <p className="font-mono text-xs uppercase tracking-widest">Loading Portfolio…</p>
      </div>
    );
  }

  if (!data || !activePortfolio || !blended) return null;

  const pieData = Object.entries(activePortfolio.weights).map(([t, w]) => ({
    name:  t,
    value: parseFloat((w * 100).toFixed(1)),
  }));

  const fmtPct = (v: number) => `${(v * 100).toFixed(2)}%`;

  return (
    <div className="space-y-6 pb-20">

      {/* ── Target Return Slider ─────────────────────────────────────────── */}
      <div className="theme-card p-8">
        <div className="flex items-center gap-3 mb-6">
          <div className="p-2 bg-accent-theme/10 rounded-xl text-accent-theme">
            <SlidersHorizontal size={18} />
          </div>
          <div>
            <h3 className="theme-label mb-0">Target Annual Return</h3>
            <p className="text-xs text-text-muted font-medium mt-0.5">
              Slide to your desired return — portfolio recommendation updates in real time
            </p>
          </div>
        </div>

        {/* Slider */}
        <div className="space-y-4">
          <div className="flex items-center justify-between text-sm font-mono">
            <span className="text-text-muted font-bold">{minReturn.toFixed(1)}%</span>
            <span className={cn("text-2xl font-bold tabular-nums tracking-tighter", recColor)}>
              {targetPct.toFixed(1)}%
            </span>
            <span className="text-text-muted font-bold">{maxReturn.toFixed(1)}%</span>
          </div>

          <div className="relative">
            <input
              type="range"
              min={minReturn}
              max={maxReturn}
              step={0.1}
              value={targetPct}
              onChange={(e) => setTargetPct(parseFloat(e.target.value))}
              className="w-full h-2 rounded-full appearance-none cursor-pointer"
              style={{
                background: `linear-gradient(to right, #4F46E5 0%, #4F46E5 ${((targetPct - minReturn) / (maxReturn - minReturn)) * 100}%, #E2E8F0 ${((targetPct - minReturn) / (maxReturn - minReturn)) * 100}%, #E2E8F0 100%)`,
              }}
            />
          </div>

          {/* Labels */}
          <div className="flex justify-between text-[10px] font-bold uppercase tracking-widest text-text-muted">
            <span className="flex items-center gap-1"><Shield size={10} /> Conservative</span>
            <span className="flex items-center gap-1"><TrendingUp size={10} /> Aggressive</span>
          </div>
        </div>

        {/* Recommendation badge */}
        <div className={cn(
          "mt-6 flex items-center justify-between px-5 py-4 rounded-2xl border",
          blend < 0.35 ? "bg-accent-theme/5 border-accent-theme/20"
            : blend > 0.65 ? "bg-neg/5 border-neg/20"
            : "bg-pos/5 border-pos/20"
        )}>
          <div>
            <p className="text-[10px] font-black uppercase tracking-widest text-text-muted">Recommended Strategy</p>
            <p className={cn("text-lg font-bold mt-0.5", recColor)}>{recommendation}</p>
          </div>
          <div className="text-right">
            <p className="text-[10px] text-text-muted font-bold uppercase tracking-widest">Blend factor</p>
            <p className="font-mono text-sm font-bold text-text-primary">{(blend * 100).toFixed(0)}% Risk-Taking</p>
          </div>
        </div>
      </div>

      {/* ── Blended metrics ──────────────────────────────────────────────── */}
      <div className="grid grid-cols-3 gap-4">
        {[
          { label: 'Projected Return', value: fmtPct(blended.expected_return), color: 'text-pos' },
          { label: 'Annual Volatility', value: fmtPct(blended.annual_volatility), color: 'text-neg' },
          { label: 'Sharpe Ratio', value: blended.sharpe_ratio.toFixed(4), color: 'text-accent-theme' },
        ].map((m) => (
          <div key={m.label} className="theme-card p-6">
            <p className="theme-label">{m.label}</p>
            <p className={cn("text-3xl font-mono font-bold tabular-nums tracking-tighter", m.color)}>{m.value}</p>
          </div>
        ))}
      </div>

      {/* ── Compare portfolios ────────────────────────────────────────────── */}
      <div className="grid grid-cols-2 gap-6">
        {(['prudent', 'risk_taking'] as const).map((key) => {
          const p = data[key];
          const isActive = (key === 'prudent' && blend < 0.5) || (key === 'risk_taking' && blend >= 0.5);
          const pd = Object.entries(p.weights).map(([t, w]) => ({ name: t, value: parseFloat((w * 100).toFixed(1)) }));
          return (
            <div key={key} className={cn("theme-card p-6 transition-all duration-300", isActive && "ring-2 ring-accent-theme/30")}>
              <div className="flex items-center justify-between mb-4">
                <div className="flex items-center gap-2">
                  {key === 'prudent' ? <Shield size={16} className="text-accent-theme" /> : <TrendingUp size={16} className="text-neg" />}
                  <p className="font-bold text-sm capitalize">{key === 'risk_taking' ? 'Risk-Taking' : 'Prudent'}</p>
                </div>
                {isActive && (
                  <span className="text-[9px] font-black uppercase tracking-widest px-2 py-1 bg-accent-theme text-white rounded-full">
                    Selected
                  </span>
                )}
              </div>

              <div className="grid grid-cols-3 gap-2 mb-4 text-center">
                {[
                  { l: 'Return', v: fmtPct(p.expected_return), c: 'text-pos' },
                  { l: 'Volatility', v: fmtPct(p.annual_volatility), c: 'text-neg' },
                  { l: 'Sharpe', v: p.sharpe_ratio.toFixed(3), c: 'text-accent-theme' },
                ].map((s) => (
                  <div key={s.l} className="bg-bg-deep rounded-xl p-2">
                    <p className="text-[9px] text-text-muted font-bold uppercase">{s.l}</p>
                    <p className={cn("text-sm font-mono font-bold tabular-nums", s.c)}>{s.v}</p>
                  </div>
                ))}
              </div>

              <div className="h-[180px]">
                <ResponsiveContainer width="100%" height="100%">
                  <PieChart>
                    <Pie data={pd} cx="50%" cy="50%" innerRadius={50} outerRadius={80} paddingAngle={3} dataKey="value">
                      {pd.map((_, i) => <Cell key={i} fill={COLORS[i % COLORS.length]} />)}
                    </Pie>
                    <Tooltip formatter={(v: number) => [`${v}%`, 'Weight']}
                      contentStyle={{ borderRadius: '12px', border: '1px solid #E2E8F0', fontSize: 11 }} />
                  </PieChart>
                </ResponsiveContainer>
              </div>

              <div className="space-y-1.5 mt-2">
                {pd.map((entry, i) => (
                  <div key={entry.name} className="flex items-center justify-between">
                    <div className="flex items-center gap-2">
                      <div className="w-2 h-2 rounded-full" style={{ backgroundColor: COLORS[i % COLORS.length] }} />
                      <span className="font-mono text-xs font-bold text-text-secondary">{entry.name}</span>
                    </div>
                    <span className="font-mono text-xs text-text-muted">{entry.value}%</span>
                  </div>
                ))}
              </div>
            </div>
          );
        })}
      </div>
    </div>
  );
};
