import { useState, useEffect } from 'react';
import { Sidebar } from './components/Sidebar';
import { PricePredictions } from './components/PricePredictions';
import { TradingSignals } from './components/TradingSignals';
import { Portfolio } from './components/Portfolio';
import { AuthModal } from './components/AuthModal';
import { AuthProvider, useAuth } from './contexts/AuthContext';
import { fetchQuote, QuoteData } from './services/marketService';
import { useLivePrices } from './hooks/useLivePrices';
import { motion, AnimatePresence } from 'motion/react';
import { AlertCircle, Loader2, LogOut } from 'lucide-react';
import { cn } from './lib/utils';

const VN_SECTOR_MAP: Record<string, string> = {
  ACB: 'Banking • HOSE',       BMC: 'Mining • HOSE',          BMP: 'Construction • HOSE',
  DHA: 'Pharma • HOSE',        DHG: 'Pharma • HOSE',          DMC: 'Pharma • HOSE',
  FMC: 'Seafood • HOSE',       FPT: 'Technology • HOSE',      GMD: 'Logistics • HOSE',
  HAX: 'Automobile • HOSE',    KDC: 'Food & Beverage • HOSE', MHC: 'Agri • HOSE',
  PAC: 'Battery • HOSE',       PET: 'Energy • HOSE',          PGC: 'Gas • HOSE',
  PPC: 'Energy • HOSE',        PVD: 'Oil & Gas • HOSE',       RAL: 'Electrical • HOSE',
  SAM: 'Telecom • HOSE',       SFI: 'Logistics • HOSE',       SJS: 'Real Estate • HOSE',
  SSI: 'Securities • HOSE',    STB: 'Banking • HOSE',         TDH: 'Real Estate • HOSE',
  VIP: 'Logistics • HOSE',     VNE: 'Construction • HOSE',    VNM: 'Food & Bev • HOSE',
};

const NASDAQ_SECTOR_MAP: Record<string, string> = {
  AAPL: 'Technology • NASDAQ',  INTC: 'Technology • NASDAQ',  TXN: 'Semiconductors • NASDAQ',
  WDC:  'Technology • NASDAQ',  DIOD: 'Semiconductors • NASDAQ', FLXS: 'Technology • NASDAQ',
  KLIC: 'Semiconductors • NASDAQ', TRNS: 'Technology • NASDAQ',
  AMGN: 'Healthcare • NASDAQ',  HELE: 'Healthcare • NASDAQ',  HOLX: 'Healthcare • NASDAQ',
  IDXX: 'Healthcare • NASDAQ',  NEOG: 'Healthcare • NASDAQ',
  CBSH: 'Financial • NASDAQ',   CINF: 'Financial • NASDAQ',   HBAN: 'Financial • NASDAQ',
  TRMK: 'Financial • NASDAQ',   UMBF: 'Financial • NASDAQ',
  JJSF: 'Consumer • NASDAQ',    MARPS: 'Consumer • NASDAQ',   MAT: 'Consumer • NASDAQ',
  MNST: 'Consumer • NASDAQ',    ROST: 'Retail • NASDAQ',      SGC: 'Consumer • NASDAQ',
  ALCO: 'Industrial • NASDAQ',  APOG: 'Industrial • NASDAQ',  ATRO: 'Industrial • NASDAQ',
  CLNE: 'Energy • NASDAQ',      MLAB: 'Industrial • NASDAQ',  OTTR: 'Utilities • NASDAQ',
  PHI: 'Industrial • NASDAQ',
};

function AppShell() {
  const { user, logout } = useAuth();

  // `market` / `ticker` = what's selected in the sidebar (may not be searched yet)
  // `currentMarket` / `currentTicker` = what was last successfully searched
  const [market, setMarket]               = useState<'NASDAQ' | 'VN'>('VN');
  const [ticker, setTicker]               = useState('FPT');
  const [currentMarket, setCurrentMarket] = useState<'NASDAQ' | 'VN'>('VN');
  const [currentTicker, setCurrentTicker] = useState('FPT');
  const [activeTab, setActiveTab]         = useState('predictions');
  const [quote, setQuote]                 = useState<QuoteData | null>(null);
  const [loading, setLoading]             = useState(false);
  const [error, setError]                 = useState<string | null>(null);
  const liveData                          = useLivePrices(currentTicker, currentMarket);

  // Accept explicit sym/mkt so Sidebar can pass values directly without
  // relying on React state having flushed yet.
  const handleSearch = async (
    overrideSym?: string,
    overrideMkt?: 'VN' | 'NASDAQ',
  ) => {
    const sym = (overrideSym ?? ticker).toUpperCase().trim();
    const mkt  = overrideMkt ?? market;
    if (!sym) return;

    // Immediately update the selected ticker so the header badge reflects
    // the user's choice without waiting for the network round-trip.
    setTicker(sym);
    setMarket(mkt);
    setLoading(true);
    setError(null);
    try {
      const data = await fetchQuote(sym, mkt);
      setQuote(data);
      setCurrentTicker(sym);
      setCurrentMarket(mkt);
    } catch {
      setError(`Could not find "${sym}" in ${mkt} market.`);
      setQuote(null);
    } finally {
      setLoading(false);
    }
  };

  // Initial load
  useEffect(() => { handleSearch(); }, []);

  // Header uses `ticker`/`market` for immediate visual feedback on selection.
  // `currentTicker`/`currentMarket` drive prediction components (only after confirmed quote).
  const displayTicker = ticker || currentTicker;
  const displayMarket = market;
  const sectorLine = displayMarket === 'VN'
    ? (VN_SECTOR_MAP[displayTicker] ?? 'Equity • HOSE • 09:00–15:00 ICT')
    : (NASDAQ_SECTOR_MAP[displayTicker] ?? 'Equity • NASDAQ • 09:30–16:00 EST');

  const renderTab = () => {
    switch (activeTab) {
      case 'predictions': return <PricePredictions ticker={currentTicker} currentPrice={quote?.regularMarketPrice ?? 0} market={currentMarket} />;
      case 'signals':     return <TradingSignals ticker={currentTicker} market={currentMarket} />;
      case 'portfolio':   return <Portfolio />;
      default:            return null;
    }
  };

  return (
    <div className="flex h-screen bg-bg-deep text-text-primary overflow-hidden font-sans">
      {/* Auth gate — show modal when logged out */}
      {!user && <AuthModal />}

      <Sidebar
        activeTab={activeTab}
        setActiveTab={setActiveTab}
        market={market}
        setMarket={setMarket}
        ticker={ticker}
        setTicker={setTicker}
        onSearch={handleSearch}
      />

      <main className="flex-1 overflow-y-auto p-8">
        <div className="max-w-6xl mx-auto w-full">

          {/* ── Header ──────────────────────────────────────────────────── */}
          <div className="flex items-end justify-between pb-8 border-b border-border-theme/60 mb-10">
            <div className="flex flex-col gap-1">
              <div className="flex items-center gap-4">
                {/* Company name from quote, or the selected ticker as immediate fallback */}
                <h2 className="text-4xl font-serif italic tracking-tight">
                  {quote?.shortName && quote.shortName !== displayTicker
                    ? quote.shortName
                    : displayTicker}
                </h2>
                <div className="flex items-center gap-2">
                  <span className="bg-text-primary text-white text-[10px] font-bold px-2 py-1 rounded tracking-widest uppercase">
                    {displayTicker}
                  </span>
                  <span className="text-[10px] font-bold text-text-muted uppercase tracking-widest">
                    {displayMarket === 'NASDAQ' ? 'US EQUITY' : 'VN EQUITY'}
                  </span>
                </div>
              </div>
              <p className="text-[11px] font-bold text-text-secondary uppercase tracking-[0.15em]">
                {sectorLine}
              </p>
            </div>

            <div className="flex items-center gap-4">
              {loading && <Loader2 className="animate-spin text-accent-theme" size={22} />}

              {/* User avatar + logout */}
              {user && (
                <div className="flex items-center gap-2">
                  <div className="w-8 h-8 rounded-full bg-accent-theme/15 border border-accent-theme/30 flex items-center justify-center">
                    <span className="text-[11px] font-black text-accent-theme uppercase">
                      {user.email[0]}
                    </span>
                  </div>
                  <button
                    onClick={logout}
                    title="Sign out"
                    className="p-1.5 rounded-lg text-text-muted hover:text-neg hover:bg-neg/10 transition-colors"
                  >
                    <LogOut size={14} />
                  </button>
                </div>
              )}

              {quote && !loading && (
                <div className="flex items-baseline gap-4">
                  <div className="text-[36px] font-mono font-medium tracking-tighter tabular-nums leading-none flex items-baseline gap-2">
                    {(() => {
                      const displayPrice = liveData?.price ?? quote.regularMarketPrice;
                      return currentMarket === 'VN'
                        ? displayPrice.toLocaleString('vi-VN')
                        : displayPrice.toLocaleString(undefined, { minimumFractionDigits: 2 });
                    })()}
                    <span className="text-xs text-text-muted font-sans font-bold uppercase tracking-widest">
                      {quote.currency}
                    </span>
                    {liveData && (
                      <span title="Live price" className="relative flex h-2 w-2 mb-1">
                        <span className="animate-ping absolute inline-flex h-full w-full rounded-full bg-pos opacity-75" />
                        <span className="relative inline-flex rounded-full h-2 w-2 bg-pos" />
                      </span>
                    )}
                  </div>
                  <div className={cn(
                    "px-3 py-1.5 rounded-lg text-sm font-bold flex items-center gap-1.5",
                    quote.regularMarketChange >= 0 ? "bg-pos/10 text-pos" : "bg-neg/10 text-neg"
                  )}>
                    {quote.regularMarketChange >= 0 ? '↑' : '↓'}
                    {' '}{Math.abs(quote.regularMarketChange).toLocaleString(undefined, { minimumFractionDigits: 2 })}
                    {' '}({quote.regularMarketChangePercent.toFixed(2)}%)
                  </div>
                </div>
              )}
            </div>
          </div>

          {error && (
            <motion.div
              initial={{ opacity: 0, y: -10 }} animate={{ opacity: 1, y: 0 }}
              className="mb-8 p-4 bg-neg/10 border border-neg/20 rounded-xl flex items-center gap-3 text-neg text-sm font-medium"
            >
              <AlertCircle size={18} />
              {error}
            </motion.div>
          )}

          <AnimatePresence mode="wait">
            <motion.div
              key={`${activeTab}-${currentTicker}-${currentMarket}`}
              initial={{ opacity: 0, x: 10 }}
              animate={{ opacity: 1, x: 0 }}
              exit={{ opacity: 0, x: -10 }}
              transition={{ duration: 0.2 }}
            >
              {renderTab()}
            </motion.div>
          </AnimatePresence>
        </div>
      </main>
    </div>
  );
}

export default function App() {
  return (
    <AuthProvider>
      <AppShell />
    </AuthProvider>
  );
}
