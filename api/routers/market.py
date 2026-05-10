"""
/api/v1/market endpoints — OHLCV history, quotes, and news.

VN tickers: reads from clean-historical-data-2026/ CSVs.
NASDAQ tickers: reads from nasdaq-historical-data/ CSVs (kept fresh by scheduler).
"""
import sys
from pathlib import Path
from datetime import datetime, timedelta

import numpy as np
import pandas as pd
from fastapi import APIRouter, HTTPException, Query

ROOT = Path(__file__).parent.parent.parent
sys.path.insert(0, str(ROOT))

router = APIRouter(prefix="/market", tags=["market"])

VN_DATA_DIR = ROOT / "clean-historical-data-2026"


# ── VN History (from CSV) ──────────────────────────────────────────────────────

def _load_vn_history(ticker: str, days: int) -> pd.DataFrame:
    path = VN_DATA_DIR / f"{ticker}_Historical.csv"
    if not path.exists():
        raise FileNotFoundError(f"No CSV for {ticker}")
    # Row 0 = column names (Date,Close,High,Low,Open,Volume)
    # Row 1 = ticker labels (,FPT.VN,...) — skip it
    # Row 2+ = data
    df = pd.read_csv(path, header=0, skiprows=[1])
    df.columns = df.columns.str.strip().str.lower()
    # Rename 'date' → 'time' for consistency
    if "date" in df.columns:
        df = df.rename(columns={"date": "time"})
    df["time"] = pd.to_datetime(df["time"])
    df = df.sort_values("time").tail(days)
    return df


# ── NASDAQ History (from local CSV) ───────────────────────────────────────────

NASDAQ_CSV_DIR = ROOT / "nasdaq-historical-data"

def _load_nasdaq_history(ticker: str, period: str) -> pd.DataFrame:
    """Read NASDAQ OHLCV from local CSV — kept fresh by the daily scheduler."""
    csv_path = NASDAQ_CSV_DIR / f"{ticker}_Historical.csv"
    if not csv_path.exists():
        raise FileNotFoundError(f"No local data for {ticker}")
    period_days = {"1mo": 22, "3mo": 66, "6mo": 132, "1y": 252}
    days = period_days.get(period, 22)
    df = pd.read_csv(csv_path)
    df.columns = [c.lower() for c in df.columns]
    df["date"] = pd.to_datetime(df["date"])
    df = df.sort_values("date").tail(days).reset_index(drop=True)
    return df


# ── Endpoints ─────────────────────────────────────────────────────────────────

@router.get("/history/{ticker}")
async def get_history(
    ticker: str,
    period: str = Query("1mo", description="1mo | 3mo | 6mo | 1y"),
    market: str = Query("VN", description="VN | NASDAQ"),
):
    """Return OHLCV bars for charting."""
    ticker = ticker.upper()
    period_days = {"1mo": 22, "3mo": 66, "6mo": 132, "1y": 252}
    days = period_days.get(period, 22)

    try:
        if market == "VN":
            df = _load_vn_history(ticker, days)
            records = [
                {
                    "date":     row["time"].strftime("%b %d"),
                    "date_iso": row["time"].strftime("%Y-%m-%d"),
                    "open":     row.get("open"),
                    "high":     row.get("high"),
                    "low":      row.get("low"),
                    "close":    row.get("close"),
                    "volume":   row.get("volume"),
                }
                for _, row in df.iterrows()
            ]
        else:
            df = _load_nasdaq_history(ticker, period)
            date_col = "date" if "date" in df.columns else df.columns[0]
            records = []
            for _, row in df.iterrows():
                raw_date = row[date_col]
                # Timezone-aware timestamps need .tz_localize(None) or just format directly
                try:
                    ts = pd.Timestamp(raw_date)
                    date_str     = ts.strftime("%b %d")
                    date_iso_str = ts.strftime("%Y-%m-%d")
                except Exception:
                    date_str     = str(raw_date)[:10]
                    date_iso_str = str(raw_date)[:10]
                records.append({
                    "date":     date_str,
                    "date_iso": date_iso_str,
                    "open":     round(float(row["open"]),  2),
                    "high":     round(float(row["high"]),  2),
                    "low":      round(float(row["low"]),   2),
                    "close":    round(float(row["close"]), 2),
                    "volume":   int(row.get("volume", 0)),
                })
    except FileNotFoundError as e:
        raise HTTPException(404, str(e))
    except Exception as e:
        raise HTTPException(500, str(e))

    return {"ticker": ticker, "market": market, "period": period, "bars": records}


@router.get("/quote/{ticker}")
async def get_quote(
    ticker: str,
    market: str = Query("VN", description="VN | NASDAQ"),
):
    """Return current quote (last close + daily change)."""
    ticker = ticker.upper()

    try:
        if market == "VN":
            df = _load_vn_history(ticker, 2)
            if len(df) < 2:
                raise HTTPException(404, "Not enough data")
            latest = df.iloc[-1]
            prev   = df.iloc[-2]
            close      = float(latest["close"])
            prev_close = float(prev["close"])
            change     = close - prev_close
            change_pct = (change / prev_close) * 100
            return {
                "ticker":                    ticker,
                "market":                    market,
                "shortName":                 ticker,
                "regularMarketPrice":        close,
                "regularMarketChange":       round(change, 2),
                "regularMarketChangePercent": round(change_pct, 2),
                "currency":                  "VND",
            }
        else:
            csv_path = NASDAQ_CSV_DIR / f"{ticker}_Historical.csv"
            if not csv_path.exists():
                raise HTTPException(404, f"No local data for {ticker}")
            df_local = pd.read_csv(csv_path)
            df_local["Date"] = pd.to_datetime(df_local["Date"])
            df_local = df_local.sort_values("Date").tail(2)
            if len(df_local) < 2:
                raise HTTPException(404, f"Insufficient data for {ticker}")
            close      = float(df_local["Close"].iloc[-1])
            prev_close = float(df_local["Close"].iloc[-2])
            change     = close - prev_close
            change_pct = (change / prev_close) * 100
            return {
                "ticker":                     ticker,
                "market":                     market,
                "shortName":                  ticker,
                "regularMarketPrice":         round(close, 2),
                "regularMarketChange":        round(change, 2),
                "regularMarketChangePercent": round(change_pct, 2),
                "currency":                   "USD",
            }
    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(500, str(e))


@router.get("/rsi/{ticker}")
async def get_rsi(
    ticker: str,
    period: int = Query(14),
    market: str = Query("VN"),
):
    """Return RSI and MA50 alongside price (for TradingSignals chart)."""
    ticker = ticker.upper()
    try:
        if market == "VN":
            df = _load_vn_history(ticker, 252)
        else:
            df = _load_nasdaq_history(ticker, "1y")
            df = df.rename(columns={"date": "time"})
    except FileNotFoundError as e:
        raise HTTPException(404, str(e))

    closes = df["close"].astype(float).values
    times  = df["time"] if "time" in df.columns else df.index

    # RSI
    deltas = np.diff(closes)
    gains  = np.where(deltas > 0, deltas, 0.0)
    losses = np.where(deltas < 0, -deltas, 0.0)
    avg_g  = pd.Series(gains).ewm(alpha=1/period, adjust=False).mean().values
    avg_l  = pd.Series(losses).ewm(alpha=1/period, adjust=False).mean().values
    rs     = np.divide(avg_g, avg_l, out=np.zeros_like(avg_g), where=avg_l != 0)
    rsi    = 100 - (100 / (1 + rs))
    rsi    = np.concatenate([[np.nan], rsi])

    # MA50
    ma50 = pd.Series(closes).rolling(50).mean().values

    records = [
        {
            "date":  t.strftime("%b %d") if hasattr(t, "strftime") else str(t),
            "price": round(float(c), 2),
            "rsi":   round(float(r), 2) if not np.isnan(r) else None,
            "ma50":  round(float(m), 2) if not np.isnan(m) else None,
        }
        for t, c, r, m in zip(times, closes, rsi, ma50)
    ]

    return {"ticker": ticker, "period": period, "data": records[-120:]}
