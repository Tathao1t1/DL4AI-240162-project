import React, { useState, useRef, useEffect, useMemo, useCallback } from 'react';
import Fuse from 'fuse.js';
import {
  LayoutDashboard, TrendingUp, Briefcase, Globe2,
  Search, X, ChevronDown, ChevronRight,
} from 'lucide-react';
import { cn } from '../lib/utils';
import type { TickerResult } from '../services/marketService';

interface SidebarProps {
  activeTab: string;
  setActiveTab: (tab: string) => void;
  market: 'NASDAQ' | 'VN';
  setMarket: (market: 'NASDAQ' | 'VN') => void;
  ticker: string;
  setTicker: (ticker: string) => void;
  onSearch: (sym?: string, mkt?: 'VN' | 'NASDAQ') => void;
}

// ── Static ticker registry (mirrors api/data/ticker_registry.json) ─────────
const ALL_TICKERS: TickerResult[] = [
  { symbol: 'ACB',  name: 'Asia Commercial Bank',               market: 'VN',     sector: 'Banking & Finance' },
  { symbol: 'SSI',  name: 'SSI Securities Corporation',         market: 'VN',     sector: 'Banking & Finance' },
  { symbol: 'STB',  name: 'Sacombank',                          market: 'VN',     sector: 'Banking & Finance' },
  { symbol: 'FPT',  name: 'FPT Corporation',                    market: 'VN',     sector: 'Technology' },
  { symbol: 'DHA',  name: 'DHA Pharmaceutical',                 market: 'VN',     sector: 'Pharma & Healthcare' },
  { symbol: 'DHG',  name: 'DHG Pharmaceutical',                 market: 'VN',     sector: 'Pharma & Healthcare' },
  { symbol: 'DMC',  name: 'Domesco Medical Import-Export',      market: 'VN',     sector: 'Pharma & Healthcare' },
  { symbol: 'KDC',  name: 'Kido Group Corporation',             market: 'VN',     sector: 'Food & Beverage' },
  { symbol: 'VNM',  name: 'Vinamilk',                           market: 'VN',     sector: 'Food & Beverage' },
  { symbol: 'PET',  name: 'PetroVietnam Technical Services',    market: 'VN',     sector: 'Energy' },
  { symbol: 'PGC',  name: 'PetroVietnam Gas (South)',           market: 'VN',     sector: 'Energy' },
  { symbol: 'PPC',  name: 'Pha Lai Thermal Power',              market: 'VN',     sector: 'Energy' },
  { symbol: 'PVD',  name: 'PetroVietnam Drilling & Well',       market: 'VN',     sector: 'Energy' },
  { symbol: 'GMD',  name: 'Gemadept Corporation',               market: 'VN',     sector: 'Logistics' },
  { symbol: 'SAM',  name: 'SAM Holdings',                       market: 'VN',     sector: 'Logistics' },
  { symbol: 'SFI',  name: 'South Logistics',                    market: 'VN',     sector: 'Logistics' },
  { symbol: 'VIP',  name: 'Vinalines Ports',                    market: 'VN',     sector: 'Logistics' },
  { symbol: 'SJS',  name: 'Song Da Urban & Industrial Zone',    market: 'VN',     sector: 'Real Estate' },
  { symbol: 'TDH',  name: 'Thu Duc Housing Development',        market: 'VN',     sector: 'Real Estate' },
  { symbol: 'BMP',  name: 'Binh Minh Plastics',                 market: 'VN',     sector: 'Construction' },
  { symbol: 'VNE',  name: 'Vietnam Electricity Construction',   market: 'VN',     sector: 'Construction' },
  { symbol: 'FMC',  name: 'Sao Ta Foods Joint Stock Company',   market: 'VN',     sector: 'Consumer Goods' },
  { symbol: 'HAX',  name: 'Hacisco',                            market: 'VN',     sector: 'Consumer Goods' },
  { symbol: 'RAL',  name: 'Rang Dong Light Source & Vacuum',    market: 'VN',     sector: 'Consumer Goods' },
  { symbol: 'BMC',  name: 'Binh Dinh Minerals',                 market: 'VN',     sector: 'Mining & Resources' },
  { symbol: 'MHC',  name: 'MHC Corporation',                    market: 'VN',     sector: 'Mining & Resources' },
  { symbol: 'PAC',  name: 'Pin Ac Quy Mien Nam',                market: 'VN',     sector: 'Mining & Resources' },
  { symbol: 'AAPL', name: 'Apple Inc.',                         market: 'NASDAQ', sector: 'Technology' },
  { symbol: 'INTC', name: 'Intel Corporation',                  market: 'NASDAQ', sector: 'Technology' },
  { symbol: 'TXN',  name: 'Texas Instruments',                  market: 'NASDAQ', sector: 'Technology' },
  { symbol: 'WDC',  name: 'Western Digital',                    market: 'NASDAQ', sector: 'Technology' },
  { symbol: 'DIOD', name: 'Diodes Incorporated',                market: 'NASDAQ', sector: 'Technology' },
  { symbol: 'FLXS', name: 'Flexsteel Industries',               market: 'NASDAQ', sector: 'Technology' },
  { symbol: 'KLIC', name: 'Kulicke and Soffa Industries',       market: 'NASDAQ', sector: 'Technology' },
  { symbol: 'TRNS', name: 'Transcat Inc.',                      market: 'NASDAQ', sector: 'Technology' },
  { symbol: 'AMGN', name: 'Amgen Inc.',                         market: 'NASDAQ', sector: 'Healthcare' },
  { symbol: 'HELE', name: 'Helen of Troy Limited',              market: 'NASDAQ', sector: 'Healthcare' },
  { symbol: 'HOLX', name: 'Hologic Inc.',                       market: 'NASDAQ', sector: 'Healthcare' },
  { symbol: 'IDXX', name: 'IDEXX Laboratories',                 market: 'NASDAQ', sector: 'Healthcare' },
  { symbol: 'NEOG', name: 'Neogen Corporation',                 market: 'NASDAQ', sector: 'Healthcare' },
  { symbol: 'CBSH', name: 'Commerce Bancshares',                market: 'NASDAQ', sector: 'Financial Services' },
  { symbol: 'CINF', name: 'Cincinnati Financial',               market: 'NASDAQ', sector: 'Financial Services' },
  { symbol: 'HBAN', name: 'Huntington Bancshares',              market: 'NASDAQ', sector: 'Financial Services' },
  { symbol: 'TRMK', name: 'Trustmark Corporation',              market: 'NASDAQ', sector: 'Financial Services' },
  { symbol: 'UMBF', name: 'UMB Financial Corporation',          market: 'NASDAQ', sector: 'Financial Services' },
  { symbol: 'JJSF', name: 'J&J Snack Foods',                    market: 'NASDAQ', sector: 'Consumer & Retail' },
  { symbol: 'MARPS',name: 'Marine Products Corporation',        market: 'NASDAQ', sector: 'Consumer & Retail' },
  { symbol: 'MAT',  name: 'Mattel Inc.',                        market: 'NASDAQ', sector: 'Consumer & Retail' },
  { symbol: 'MNST', name: 'Monster Beverage Corporation',       market: 'NASDAQ', sector: 'Consumer & Retail' },
  { symbol: 'ROST', name: 'Ross Stores',                        market: 'NASDAQ', sector: 'Consumer & Retail' },
  { symbol: 'SGC',  name: 'Superior Group of Companies',        market: 'NASDAQ', sector: 'Consumer & Retail' },
  { symbol: 'ALCO', name: 'Alico Inc.',                         market: 'NASDAQ', sector: 'Industrial' },
  { symbol: 'APOG', name: 'Apogee Enterprises',                 market: 'NASDAQ', sector: 'Industrial' },
  { symbol: 'ATRO', name: 'Astronics Corporation',              market: 'NASDAQ', sector: 'Industrial' },
  { symbol: 'CLNE', name: 'Clean Energy Fuels',                 market: 'NASDAQ', sector: 'Industrial' },
  { symbol: 'MLAB', name: 'Mesa Labs',                          market: 'NASDAQ', sector: 'Industrial' },
  { symbol: 'OTTR', name: 'Otter Tail Corporation',             market: 'NASDAQ', sector: 'Industrial' },
  { symbol: 'PHI',  name: 'PLDT Inc. (Philippines)',            market: 'NASDAQ', sector: 'Industrial' },
];

const VN_SECTORS: Record<string, string[]> = {
  'Technology':          ['FPT'],
  'Banking & Finance':   ['ACB', 'SSI', 'STB'],
  'Pharma & Healthcare': ['DHA', 'DHG', 'DMC'],
  'Food & Beverage':     ['KDC', 'VNM'],
  'Energy':              ['PET', 'PGC', 'PPC', 'PVD'],
  'Logistics':           ['GMD', 'SAM', 'SFI', 'VIP'],
  'Real Estate':         ['SJS', 'TDH'],
  'Construction':        ['BMP', 'VNE'],
  'Consumer Goods':      ['FMC', 'HAX', 'RAL'],
  'Mining & Resources':  ['BMC', 'MHC', 'PAC'],
};

const NASDAQ_SECTORS: Record<string, string[]> = {
  'Technology':         ['AAPL', 'INTC', 'TXN', 'WDC', 'DIOD', 'FLXS', 'KLIC', 'TRNS'],
  'Healthcare':         ['AMGN', 'HELE', 'HOLX', 'IDXX', 'NEOG'],
  'Financial Services': ['CBSH', 'CINF', 'HBAN', 'TRMK', 'UMBF'],
  'Consumer & Retail':  ['JJSF', 'MARPS', 'MAT', 'MNST', 'ROST', 'SGC'],
  'Industrial':         ['ALCO', 'APOG', 'ATRO', 'CLNE', 'MLAB', 'OTTR', 'PHI'],
};

const fuse = new Fuse(ALL_TICKERS, {
  keys: ['symbol', 'name'],
  threshold: 0.35,
  includeScore: true,
});

export const Sidebar: React.FC<SidebarProps> = ({
  activeTab, setActiveTab, market, setMarket, ticker, setTicker, onSearch,
}) => {
  const [query, setQuery]         = useState('');
  const [activeIdx, setActiveIdx] = useState(-1);
  const [browseOpen, setBrowseOpen] = useState(false);
  const inputRef                  = useRef<HTMLInputElement>(null);
  const listRef                   = useRef<HTMLUListElement>(null);

  const results: TickerResult[] = useMemo(() => {
    if (!query.trim()) return [];
    return fuse.search(query.trim()).slice(0, 10).map(r => r.item);
  }, [query]);

  // Reset active index when results change
  useEffect(() => { setActiveIdx(-1); }, [results]);

  const handleSelect = useCallback((t: TickerResult) => {
    setQuery('');
    setTicker(t.symbol);
    if (t.market !== market) setMarket(t.market);
    onSearch(t.symbol, t.market);
  }, [market, setMarket, setTicker, onSearch]);

  const handleKeyDown = (e: React.KeyboardEvent) => {
    if (!results.length) return;
    if (e.key === 'ArrowDown') {
      e.preventDefault();
      setActiveIdx(i => Math.min(i + 1, results.length - 1));
    } else if (e.key === 'ArrowUp') {
      e.preventDefault();
      setActiveIdx(i => Math.max(i - 1, -1));
    } else if (e.key === 'Enter' && activeIdx >= 0) {
      e.preventDefault();
      handleSelect(results[activeIdx]);
    } else if (e.key === 'Escape') {
      setQuery('');
    }
  };

  // Scroll active item into view
  useEffect(() => {
    if (activeIdx >= 0 && listRef.current) {
      const el = listRef.current.children[activeIdx] as HTMLElement;
      el?.scrollIntoView({ block: 'nearest' });
    }
  }, [activeIdx]);

  const handleMarketSwitch = (m: 'VN' | 'NASDAQ') => {
    if (m === market) return;
    const newSectors = m === 'VN' ? VN_SECTORS : NASDAQ_SECTORS;
    const firstTicker = Object.values(newSectors)[0][0];
    setMarket(m);
    setTicker(firstTicker);
    onSearch(firstTicker, m);
  };

  const sectors = market === 'VN' ? VN_SECTORS : NASDAQ_SECTORS;

  const tabs = [
    { id: 'predictions', label: 'Price Predictions', icon: TrendingUp },
    { id: 'signals',     label: 'Trading Signals',   icon: LayoutDashboard },
    { id: 'portfolio',   label: 'Portfolio Manager', icon: Briefcase },
  ];

  return (
    <div className="w-[300px] h-screen bg-bg-sidebar border-r border-border-theme flex flex-col p-8 text-text-primary">
      {/* Logo */}
      <div className="mb-12">
        <h1 className="text-xl font-bold tracking-tighter flex items-center gap-3">
          <div className="w-9 h-9 bg-accent-theme rounded-xl flex items-center justify-center shadow-lg shadow-accent-theme/20">
            <TrendingUp size={20} className="text-white" />
          </div>
          QuantPulse AI
        </h1>
        <p className="text-[10px] uppercase tracking-[0.2em] text-text-muted mt-3 font-bold">
          Financial Intelligence
        </p>
      </div>

      <div className="space-y-8 flex-1 min-h-0 overflow-y-auto">
        {/* Market toggle */}
        <div>
          <label className="theme-label flex items-center gap-2">
            <Globe2 size={12} /> Market Workspace
          </label>
          <div className="grid grid-cols-2 gap-2 p-1 bg-white/50 border border-border-theme rounded-xl">
            {(['VN', 'NASDAQ'] as const).map((m) => (
              <button key={m} onClick={() => handleMarketSwitch(m)}
                className={cn(
                  "py-2 px-3 text-[11px] rounded-lg transition-all font-bold tracking-wide",
                  market === m
                    ? "bg-white text-accent-theme shadow-sm border border-border-theme"
                    : "text-text-muted hover:text-text-secondary"
                )}>
                {m === 'VN' ? 'VIETNAM' : 'NASDAQ'}
              </button>
            ))}
          </div>
        </div>

        {/* Search */}
        <div>
          <label className="theme-label">Search Ticker</label>

          {/* Input */}
          <div className="relative">
            <Search size={14} className="absolute left-3 top-1/2 -translate-y-1/2 text-text-muted pointer-events-none" />
            <input
              ref={inputRef}
              type="text"
              value={query}
              onChange={e => setQuery(e.target.value)}
              onKeyDown={handleKeyDown}
              placeholder="Search symbol or company…"
              className="theme-input w-full pl-8 pr-8 text-sm font-mono"
              autoComplete="off"
              spellCheck={false}
            />
            {query && (
              <button
                onClick={() => { setQuery(''); inputRef.current?.focus(); }}
                className="absolute right-3 top-1/2 -translate-y-1/2 text-text-muted hover:text-text-primary transition-colors"
              >
                <X size={13} />
              </button>
            )}
          </div>

          {/* Results dropdown */}
          {results.length > 0 && (
            <ul
              ref={listRef}
              className="mt-1.5 bg-bg-surface border border-border-theme rounded-2xl shadow-xl overflow-hidden max-h-60 overflow-y-auto"
            >
              {results.map((r, i) => (
                <li key={r.symbol}>
                  <button
                    onClick={() => handleSelect(r)}
                    className={cn(
                      "w-full text-left px-4 py-2.5 transition-colors flex items-center justify-between gap-2",
                      i === activeIdx
                        ? "bg-accent-theme/10 text-accent-theme"
                        : "text-text-secondary hover:bg-bg-sidebar hover:text-text-primary"
                    )}
                  >
                    <span className="flex items-center gap-2 min-w-0">
                      <span className="font-mono font-bold text-sm shrink-0">{r.symbol}</span>
                      <span className="text-[10px] text-text-muted truncate">{r.name}</span>
                    </span>
                    <span className={cn(
                      "text-[9px] font-black uppercase tracking-widest px-1.5 py-0.5 rounded shrink-0",
                      r.market === 'VN'
                        ? "bg-accent-theme/10 text-accent-theme"
                        : "bg-pos/10 text-pos"
                    )}>
                      {r.market}
                    </span>
                  </button>
                </li>
              ))}
            </ul>
          )}

          {/* Active ticker pill */}
          {ticker && !query && (
            <div className="mt-2 flex items-center gap-2 px-3 py-2 bg-accent-theme/5 border border-accent-theme/15 rounded-xl">
              <span className="font-mono font-black text-sm text-accent-theme">{ticker}</span>
              <span className="text-[10px] text-text-muted truncate flex-1">
                {ALL_TICKERS.find(t => t.symbol === ticker)?.name ?? ''}
              </span>
            </div>
          )}

          <button onClick={() => onSearch(ticker, market)} disabled={!ticker}
            className="mt-3 w-full theme-btn-primary flex items-center justify-center gap-2">
            <TrendingUp size={15} />
            Predict
          </button>
        </div>

        {/* Browse by sector (collapsible) */}
        <div>
          <button
            onClick={() => setBrowseOpen(o => !o)}
            className="theme-label flex items-center gap-2 w-full hover:text-text-secondary transition-colors"
          >
            {browseOpen ? <ChevronDown size={12} /> : <ChevronRight size={12} />}
            Browse by Sector
          </button>

          {browseOpen && (
            <div className="mt-2 border border-border-theme rounded-2xl overflow-hidden max-h-52 overflow-y-auto">
              {Object.entries(sectors).map(([sector, tickers]) => (
                <div key={sector}>
                  <div className="px-4 py-2 text-[9px] font-black uppercase tracking-[0.2em] text-text-muted bg-bg-deep border-b border-border-theme/50 sticky top-0">
                    {sector}
                  </div>
                  {tickers.map((t) => (
                    <button key={t}
                      onClick={() => { setTicker(t); onSearch(t, market); }}
                      className={cn(
                        "w-full text-left px-4 py-2 text-sm font-mono font-semibold transition-colors",
                        ticker === t
                          ? "bg-accent-theme/10 text-accent-theme"
                          : "text-text-secondary hover:bg-bg-sidebar hover:text-text-primary"
                      )}>
                      {t}
                    </button>
                  ))}
                </div>
              ))}
            </div>
          )}
        </div>

        {/* Navigation */}
        <nav>
          <label className="theme-label">Navigation</label>
          <div className="space-y-1">
            {tabs.map((tab) => (
              <button key={tab.id} onClick={() => setActiveTab(tab.id)}
                className={cn(
                  "w-full flex items-center gap-3 px-4 py-3.5 rounded-xl text-sm transition-all font-semibold",
                  activeTab === tab.id
                    ? "bg-white text-accent-theme shadow-sm border border-border-theme"
                    : "text-text-secondary hover:text-text-primary hover:bg-white/50"
                )}>
                <tab.icon size={18} className={activeTab === tab.id ? "text-accent-theme" : "text-text-muted"} />
                {tab.label}
              </button>
            ))}
          </div>
        </nav>
      </div>

      {/* Status */}
      <div className="mt-auto pt-8 border-t border-border-theme/60 shrink-0">
        <div className="flex items-center gap-3 p-4 bg-pos/5 rounded-2xl border border-pos/10">
          <div className="relative flex h-2 w-2">
            <span className="animate-ping absolute inline-flex h-full w-full rounded-full bg-pos opacity-75" />
            <span className="relative inline-flex rounded-full h-2 w-2 bg-pos" />
          </div>
          <div>
            <div className="text-[10px] font-bold text-pos uppercase tracking-wider">Engine Active</div>
            <div className="text-[9px] text-text-muted font-mono">v4.2.0-stable</div>
          </div>
        </div>
      </div>
    </div>
  );
};
