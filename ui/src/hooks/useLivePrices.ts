/**
 * useLivePrices — subscribes to the SSE /live/prices stream for one ticker.
 *
 * Passes the market so the backend skips the .VN suffix for NASDAQ tickers.
 * Returns the latest live price (null until the first event arrives).
 * Closes the EventSource automatically on ticker/market change or unmount.
 */
import { useEffect, useRef, useState } from 'react';

interface LivePrice {
  ticker: string;
  market: string;
  price: number;
  change_pct: number;
  ts: string;
}

export function useLivePrices(
  ticker: string,
  market: 'VN' | 'NASDAQ',
): LivePrice | null {
  const [liveData, setLiveData] = useState<LivePrice | null>(null);
  const esRef = useRef<EventSource | null>(null);

  useEffect(() => {
    if (!ticker) return;

    // Close previous connection
    if (esRef.current) {
      esRef.current.close();
      esRef.current = null;
    }

    setLiveData(null);

    const url = `/api/v1/live/prices?tickers=${encodeURIComponent(ticker)}&market=${market}`;
    const es  = new EventSource(url);
    esRef.current = es;

    es.onmessage = (e) => {
      try {
        const d = JSON.parse(e.data) as LivePrice;
        if (d.ticker === ticker) setLiveData(d);
      } catch {
        // ignore malformed events
      }
    };

    es.onerror = () => {
      // Browser auto-reconnects on transient failures
    };

    return () => {
      es.close();
      esRef.current = null;
    };
  }, [ticker, market]);

  return liveData;
}
