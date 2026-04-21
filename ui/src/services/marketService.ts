/**
 * marketService.ts — all API calls to the FastAPI backend.
 *
 * In development: Vite proxy forwards /api/* → http://localhost:8000
 * In production:  nginx rewrites /api/* → fastapi container
 */

const BASE = '/api/v1';

async function get<T>(path: string): Promise<T> {
  const res = await fetch(`${BASE}${path}`);
  if (!res.ok) throw new Error(`API ${path} → ${res.status} ${res.statusText}`);
  return res.json() as Promise<T>;
}

// ── Types ──────────────────────────────────────────────────────────────────────

export interface QuoteData {
  ticker: string;
  market: string;
  shortName: string;
  regularMarketPrice: number;
  regularMarketChange: number;
  regularMarketChangePercent: number;
  currency: string;
}

export interface BarData {
  date: string;
  date_iso?: string;
  open?: number;
  high?: number;
  low?: number;
  close?: number;
  volume?: number;
}

export interface RSIData {
  date: string;
  price: number;
  rsi: number | null;
  ma50: number | null;
}

export interface SignalData {
  ticker: string;
  buy_prob: number;
  sell_prob: number;
  threshold: number;
  signal: 'BUY' | 'SELL' | 'HOLD';
  // Task 3 model performance metrics (from task3_buy/sell_metrics.csv)
  buy_auc: number;
  buy_f1: number;
  buy_precision: number;
  buy_recall: number;
  sell_auc: number;
  sell_f1: number;
  sell_precision: number;
  sell_recall: number;
}

export interface PredictionData {
  ticker: string;
  task: string;
  last_close: number;
  predictions: Record<string, number>;
  unit: string;
  run_at?: string;
}

export interface PortfolioData {
  label: string;
  expected_return: number;
  annual_volatility: number;
  sharpe_ratio: number;
  rf_rate: number;
  weights: Record<string, number>;
  tickers: string[];
  n_stocks: number;
}

// ── Endpoints ─────────────────────────────────────────────────────────────────

/** Current quote for one ticker */
export async function fetchQuote(ticker: string, market: 'VN' | 'NASDAQ'): Promise<QuoteData> {
  return get<QuoteData>(`/market/quote/${ticker}?market=${market}`);
}

/** OHLCV bars for price chart */
export async function fetchStockHistory(
  ticker: string,
  period: '1mo' | '3mo' | '6mo' | '1y',
  market: 'VN' | 'NASDAQ',
): Promise<BarData[]> {
  const data = await get<{ bars: BarData[] }>(`/market/history/${ticker}?period=${period}&market=${market}`);
  return data.bars;
}

/** RSI + MA50 + price for TradingSignals chart */
export async function fetchRSIData(ticker: string, market: 'VN' | 'NASDAQ'): Promise<RSIData[]> {
  const data = await get<{ data: RSIData[] }>(`/market/rsi/${ticker}?market=${market}`);
  return data.data;
}

/** Task 3 buy/sell signal probabilities */
export async function fetchSignal(ticker: string): Promise<SignalData> {
  return get<SignalData>(`/signals/${ticker}`);
}

/** Task 2 price prediction (VN tickers) */
export async function fetchPrediction(ticker: string, task: string): Promise<PredictionData> {
  return get<PredictionData>(`/predict/${ticker}?task=${task}`);
}

/** Task 1 price prediction (NASDAQ tickers). k=1 next-day, k=3 3rd-day, k=7 7th-day */
export async function fetchNasdaqPrediction(ticker: string, k: 1 | 3 | 7 = 1): Promise<PredictionData> {
  return get<PredictionData>(`/predict/nasdaq/${ticker}?k=${k}`);
}

/** Task 4 portfolio summary (both prudent + risk-taking) */
export async function fetchPortfolio(): Promise<{ prudent: PortfolioData; risk_taking: PortfolioData }> {
  return get<{ prudent: PortfolioData; risk_taking: PortfolioData }>('/portfolio/summary');
}

/** Ticker search result */
export interface TickerResult {
  symbol: string;
  name: string;
  market: 'VN' | 'NASDAQ';
  sector: string;
}

/** Fuzzy ticker search */
export async function searchTickers(q: string): Promise<TickerResult[]> {
  return get<TickerResult[]>(`/search?q=${encodeURIComponent(q)}`);
}
