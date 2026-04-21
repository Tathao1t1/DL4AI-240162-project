"""
/api/v1/live/prices — Server-Sent Events stream for real-time quotes.

GET /live/prices?tickers=FPT,ACB&market=VN
GET /live/prices?tickers=AAPL,INTC&market=NASDAQ

→ text/event-stream pushing { ticker, price, change_pct, ts } every 30 s.

Market-aware:
  VN     → tries {symbol}.VN first, falls back to plain symbol
  NASDAQ → uses plain symbol directly (skips the costly .VN 404 round-trip)

yfinance calls run in asyncio.to_thread() to avoid blocking the event loop.
"""
import asyncio
import json
import logging
from datetime import datetime, timezone

from fastapi import APIRouter, Query, Request
from fastapi.responses import StreamingResponse

logger = logging.getLogger("live")

# yfinance logs ERROR for every "symbol may be delisted" hit — these are
# expected when we probe {symbol}.VN and it doesn't exist, so silence them.
logging.getLogger("yfinance").setLevel(logging.CRITICAL)

router = APIRouter(prefix="/live", tags=["live"])


def _fetch_yf_price(symbol: str, market: str = "VN") -> dict:
    """
    Blocking yfinance fetch — runs in a thread pool.

    For VN:     tries {symbol}.VN → falls back to plain symbol
    For NASDAQ: uses plain symbol directly (no .VN prefix attempt)
    """
    import yfinance as yf

    candidates = [f"{symbol}.VN", symbol] if market == "VN" else [symbol]

    for sym in candidates:
        try:
            fi    = yf.Ticker(sym).fast_info
            price = fi.last_price
            prev  = fi.previous_close
            if price and float(price) > 0:
                change_pct = ((float(price) - float(prev)) / float(prev) * 100) if prev else 0.0
                return {
                    "price":      round(float(price), 4),
                    "change_pct": round(change_pct, 4),
                }
        except Exception:
            continue

    raise ValueError(f"No live price data for {symbol} (market={market})")


@router.get("/prices")
async def stream_prices(
    request: Request,
    tickers: str = Query(..., description="Comma-separated list, e.g. FPT,ACB or AAPL,INTC"),
    market:  str = Query("VN", description="VN | NASDAQ"),
):
    """SSE stream — emits one JSON event per ticker every 30 seconds."""
    symbols = [t.strip().upper() for t in tickers.split(",") if t.strip()]
    mkt     = market.upper()

    async def generator():
        while True:
            # Stop immediately if the client disconnected (no more orphaned generators)
            if await request.is_disconnected():
                logger.debug("SSE client disconnected — stopping generator for %s (%s)", symbols, mkt)
                return

            for sym in symbols:
                try:
                    data = await asyncio.to_thread(_fetch_yf_price, sym, mkt)
                    payload = json.dumps({
                        "ticker": sym,
                        "market": mkt,
                        **data,
                        "ts": datetime.now(timezone.utc).isoformat(),
                    })
                    yield f"data: {payload}\n\n"
                except Exception as e:
                    logger.debug("live price failed %s (%s): %s", sym, mkt, e)

            # Sleep in small increments so disconnect is detected within 2 s
            for _ in range(15):
                if await request.is_disconnected():
                    return
                await asyncio.sleep(2)

    return StreamingResponse(
        generator(),
        media_type="text/event-stream",
        headers={
            "Cache-Control":     "no-cache",
            "X-Accel-Buffering": "no",
            "Connection":        "keep-alive",
        },
    )
