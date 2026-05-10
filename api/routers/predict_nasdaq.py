"""
/api/v1/predict/nasdaq/{ticker} — Task 1 NASDAQ price predictions.

Model location: models/task1_1/next_day/per_ticker/{TICKER}/
Features: 20 ratio-based technical indicators (see metadata.json).

Data strategy:
  - Local CSVs in nasdaq-historical-data/ are read on every request (fast).
  - If the CSV is missing, data is downloaded from yfinance and saved (first run).
  - The daily scheduler keeps CSVs fresh; a live top-up is also done on each call.
"""
import logging, pickle, sys
from pathlib import Path

import numpy as np
import pandas as pd
from fastapi import APIRouter, HTTPException

ROOT = Path(__file__).parent.parent.parent
sys.path.insert(0, str(ROOT))

logger = logging.getLogger("predict_nasdaq")

router = APIRouter(prefix="/predict/nasdaq", tags=["predict-nasdaq"])

TASK1_DIR       = ROOT / "models" / "task1_1" / "next_day" / "per_ticker"
TASK1_2_DIR     = ROOT / "models" / "task1_2"
TASK1_3_DIR     = ROOT / "models" / "task1_3"
NASDAQ_DATA_DIR = ROOT / "nasdaq-historical-data"
NASDAQ_TICKERS  = sorted([p.name for p in TASK1_DIR.iterdir() if p.is_dir()])

# In-memory model caches
# _cache: (ticker, k) → (model, scaler_X, scaler_y) for task1_1 and task1_2
# _cache_consecutive: (ticker, k) → (model, scaler_X, scaler_y) for task1_3
_cache: dict = {}
_cache_consecutive: dict = {}


def _load_model_into_cache(ticker: str, k: int):
    """Load task1_1/task1_2 model + scalers for (ticker, k) into _cache."""
    import tensorflow as tf
    if k == 1:
        model_dir = TASK1_DIR / ticker
    elif k in (3, 7):
        model_dir = TASK1_2_DIR / f"k{k}" / "per_ticker" / ticker
    else:
        return
    if not model_dir.exists():
        return
    with open(model_dir / "scaler_X.pkl", "rb") as f:
        scaler_X = pickle.load(f)
    with open(model_dir / "scaler_y.pkl", "rb") as f:
        scaler_y = pickle.load(f)
    model = tf.keras.models.load_model(model_dir / "model.keras", compile=False)
    _cache[(ticker, k)] = (model, scaler_X, scaler_y)


def _load_consecutive_model_into_cache(ticker: str, k: int):
    """Load task1_3 consecutive model + scalers for (ticker, k) into _cache_consecutive."""
    import tensorflow as tf
    model_dir = TASK1_3_DIR / f"k{k}" / "per_ticker" / ticker
    if not model_dir.exists():
        return
    with open(model_dir / "scaler_X.pkl", "rb") as f:
        scaler_X = pickle.load(f)
    with open(model_dir / "scaler_y.pkl", "rb") as f:
        scaler_y = pickle.load(f)
    model = tf.keras.models.load_model(model_dir / "model.keras", compile=False)
    _cache_consecutive[(ticker, k)] = (model, scaler_X, scaler_y)


async def preload_all():
    """Warm all model caches at startup (task1_1, task1_2, task1_3)."""
    import asyncio
    loop = asyncio.get_event_loop()
    for ticker in NASDAQ_TICKERS:
        for k in (1, 3, 7):
            if (ticker, k) not in _cache:
                try:
                    await loop.run_in_executor(None, _load_model_into_cache, ticker, k)
                except Exception as e:
                    logger.warning("preload failed %s k=%d: %s", ticker, k, e)
        for k in (3, 7):
            if (ticker, k) not in _cache_consecutive:
                try:
                    await loop.run_in_executor(None, _load_consecutive_model_into_cache, ticker, k)
                except Exception as e:
                    logger.warning("preload consecutive failed %s k=%d: %s", ticker, k, e)


# ── Local CSV management ───────────────────────────────────────────────────────

def _load_nasdaq_csv(ticker: str) -> pd.DataFrame:
    """
    Return a DataFrame with columns [Date, Open, High, Low, Close, Volume].
    - If the CSV exists: read it and top-up any missing rows from yfinance.
    - If the CSV is missing: download 2 years from yfinance and save it.
    Date column is always returned as datetime.
    """
    import yfinance as yf

    NASDAQ_DATA_DIR.mkdir(exist_ok=True)
    csv_path = NASDAQ_DATA_DIR / f"{ticker}_Historical.csv"

    if csv_path.exists():
        df = pd.read_csv(csv_path)
        df["Date"] = pd.to_datetime(df["Date"])
        last_date  = df["Date"].iloc[-1]

        # Top-up: fetch the last 30 days and append any new rows
        try:
            raw = yf.Ticker(ticker).history(period="1mo", interval="1d")
            if not raw.empty:
                raw = raw.reset_index()
                raw["Date"] = pd.to_datetime(raw["Date"]).dt.tz_localize(None)
                new_rows = raw[raw["Date"] > last_date].copy()
                if not new_rows.empty:
                    new_rows = new_rows[["Date", "Open", "High", "Low", "Close", "Volume"]].copy()
                    new_rows["Date"] = new_rows["Date"].dt.strftime("%Y-%m-%d")
                    df["Date"] = df["Date"].dt.strftime("%Y-%m-%d")
                    df = (
                        pd.concat([df, new_rows])
                        .drop_duplicates(subset="Date", keep="last")
                        .sort_values("Date")
                        .reset_index(drop=True)
                    )
                    df.to_csv(csv_path, index=False)
                    logger.info("nasdaq csv %s: +%d new rows", ticker, len(new_rows))
                    df["Date"] = pd.to_datetime(df["Date"])
        except Exception as e:
            logger.debug("nasdaq top-up %s failed: %s", ticker, e)

        from api.utils.validation import validate_ohlcv
        return validate_ohlcv(df, ticker)

    # ── First run: bootstrap with 2 years of history ──────────────────────────
    logger.info("nasdaq csv %s: not found — bootstrapping from yfinance", ticker)
    raw = yf.download(ticker, period="2y", interval="1d", progress=False, auto_adjust=True)
    if raw.empty:
        raise ValueError(f"yfinance returned no data for {ticker}")
    raw = raw.reset_index()
    raw.columns = [c[0] if isinstance(c, tuple) else c for c in raw.columns]
    raw["Date"] = pd.to_datetime(raw["Date"]).dt.tz_localize(None)
    df = raw[["Date", "Open", "High", "Low", "Close", "Volume"]].copy()
    df["Date"] = df["Date"].dt.strftime("%Y-%m-%d")
    df.to_csv(csv_path, index=False)
    logger.info("nasdaq csv %s: bootstrapped with %d rows", ticker, len(df))
    df["Date"] = pd.to_datetime(df["Date"])
    from api.utils.validation import validate_ohlcv
    return validate_ohlcv(df, ticker)


def _add_nasdaq_features(df: pd.DataFrame) -> pd.DataFrame:
    """Compute the 20 ratio-based features used by Task 1 models."""
    c = df["Close"].astype(float)
    v = df["Volume"].astype(float)

    df["Return_1d"]       = c.pct_change(1)
    df["Return_5d"]       = c.pct_change(5)
    df["Return_10d"]      = c.pct_change(10)
    df["Return_20d"]      = c.pct_change(20)

    sma10  = c.rolling(10).mean()
    sma20  = c.rolling(20).mean()
    sma50  = c.rolling(50).mean()
    ema10  = c.ewm(span=10, adjust=False).mean()
    ema20  = c.ewm(span=20, adjust=False).mean()

    df["SMA10_vs_SMA20"]  = (sma10  - sma20)  / sma20
    df["SMA20_vs_SMA50"]  = (sma20  - sma50)  / sma50
    df["EMA10_vs_EMA20"]  = (ema10  - ema20)  / ema20
    df["Close_vs_SMA20"]  = (c      - sma20)  / sma20
    df["Close_vs_SMA50"]  = (c      - sma50)  / sma50

    # RSI-14
    delta  = c.diff()
    gain   = delta.clip(lower=0).ewm(alpha=1/14, adjust=False).mean()
    loss   = (-delta.clip(upper=0)).ewm(alpha=1/14, adjust=False).mean()
    rs     = gain / loss.replace(0, np.nan)
    df["RSI_14"] = 100 - 100 / (1 + rs)

    # MACD
    ema12  = c.ewm(span=12, adjust=False).mean()
    ema26  = c.ewm(span=26, adjust=False).mean()
    macd   = ema12 - ema26
    sig    = macd.ewm(span=9, adjust=False).mean()
    df["MACD_pct"]      = macd / c
    df["MACD_sig_pct"]  = sig  / c
    df["MACD_hist_pct"] = (macd - sig) / c

    # Bollinger Bands
    bb_mid = sma20
    bb_std = c.rolling(20).std()
    df["BB_width"] = (2 * bb_std) / bb_mid
    df["BB_pct"]   = (c - (bb_mid - 2 * bb_std)) / (4 * bb_std)

    # ATR-14
    high = df["High"].astype(float)
    low  = df["Low"].astype(float)
    tr   = pd.concat([high - low,
                       (high - c.shift()).abs(),
                       (low  - c.shift()).abs()], axis=1).max(axis=1)
    df["ATR_pct"] = tr.ewm(span=14, adjust=False).mean() / c

    df["Volatility_10"]   = c.pct_change().rolling(10).std()
    df["Volatility_20"]   = c.pct_change().rolling(20).std()
    df["Volume_ratio"]    = v / v.rolling(20).mean()
    df["High_Low_pct"]    = (high - low) / c

    return df


FEATURE_COLS = [
    "Return_1d", "Return_5d", "Return_10d", "Return_20d",
    "SMA10_vs_SMA20", "SMA20_vs_SMA50", "EMA10_vs_EMA20",
    "Close_vs_SMA20", "Close_vs_SMA50",
    "RSI_14", "MACD_pct", "MACD_sig_pct", "MACD_hist_pct",
    "BB_width", "BB_pct", "ATR_pct",
    "Volatility_10", "Volatility_20", "Volume_ratio", "High_Low_pct",
]


def _run_inference(ticker: str, k: int) -> dict:
    """
    Core inference shared by all three NASDAQ horizons (k=1, 3, 7).
    k=1 → task1_1/next_day/per_ticker/{ticker}/
    k=3 → task1_2/k3/per_ticker/{ticker}/
    k=7 → task1_2/k7/per_ticker/{ticker}/
    """
    import tensorflow as tf

    ticker = ticker.upper()
    if k == 1:
        model_dir = TASK1_DIR / ticker
        task_label = "task1_1"
    elif k in (3, 7):
        model_dir = TASK1_2_DIR / f"k{k}" / "per_ticker" / ticker
        task_label = f"task1_2_k{k}"
    else:
        raise ValueError(f"Unsupported k={k}; must be 1, 3, or 7")

    cache_key = (ticker, k)
    if cache_key not in _cache:
        _load_model_into_cache(ticker, k)
    if cache_key not in _cache:
        raise ValueError(f"No model for {ticker} (k={k})")

    model, scaler_X, scaler_y = _cache[cache_key]

    df = _load_nasdaq_csv(ticker)
    if df.empty:
        raise ValueError(f"No data for {ticker}")

    if "Date" not in df.columns and "date" in df.columns:
        df = df.rename(columns={"date": "Date"})

    df = _add_nasdaq_features(df)
    df = df.dropna(subset=FEATURE_COLS)

    lookback = 60
    X = df[FEATURE_COLS].values[-lookback:]
    if len(X) < lookback:
        raise ValueError(f"Not enough data for {ticker} (got {len(X)}, need {lookback})")

    X_scaled   = scaler_X.transform(X).reshape(1, lookback, len(FEATURE_COLS))
    y_scaled   = model.predict(X_scaled, verbose=0)
    log_return = float(scaler_y.inverse_transform(y_scaled)[0][0])

    last_close = float(df["Close"].iloc[-1])
    last_date  = str(df["Date"].iloc[-1])[:10] if "Date" in df.columns else "N/A"

    # Models predict log returns: reconstruct price with exp, never fall back to
    # the raw scaler output (which is not a price).
    pred_price = last_close * float(np.exp(log_return))
    day_key    = f"day_{k}"

    return {
        "ticker":      ticker,
        "task":        task_label,
        "market":      "NASDAQ",
        "last_date":   last_date,
        "last_close":  round(last_close, 2),
        "predictions": {day_key: round(pred_price, 2)},
        "unit":        "USD",
    }


def _run_inference_consecutive(ticker: str, k: int) -> dict:
    """
    Task 1.3: predict k consecutive daily prices for a NASDAQ ticker.
    Returns predictions = {day_1: price, day_2: price, ..., day_k: price}
    using cumulative log-return reconstruction: price_j = last_close * exp(Σ lr_1..j)
    """
    import tensorflow as tf

    ticker = ticker.upper()
    cache_key = (ticker, k)
    if cache_key not in _cache_consecutive:
        _load_consecutive_model_into_cache(ticker, k)
    if cache_key not in _cache_consecutive:
        raise ValueError(f"No task1_3 model for {ticker} (k={k})")

    model, scaler_X, scaler_y = _cache_consecutive[cache_key]

    df = _load_nasdaq_csv(ticker)
    if df.empty:
        raise ValueError(f"No data for {ticker}")

    if "Date" not in df.columns and "date" in df.columns:
        df = df.rename(columns={"date": "Date"})

    df = _add_nasdaq_features(df)
    df = df.dropna(subset=FEATURE_COLS)

    lookback = 60
    X = df[FEATURE_COLS].values[-lookback:]
    if len(X) < lookback:
        raise ValueError(f"Not enough data for {ticker} (got {len(X)}, need {lookback})")

    X_scaled = scaler_X.transform(X).reshape(1, lookback, len(FEATURE_COLS))
    y_scaled = model.predict(X_scaled, verbose=0)          # shape (1, k)
    log_rets = scaler_y.inverse_transform(y_scaled.reshape(-1, 1)).flatten()

    last_close = float(df["Close"].iloc[-1])
    last_date  = str(df["Date"].iloc[-1])[:10] if "Date" in df.columns else "N/A"

    # Cumulative log-returns → price for each future day
    predictions: dict = {}
    cumsum = 0.0
    for j, lr in enumerate(log_rets):
        cumsum += float(lr)
        predictions[f"day_{j + 1}"] = round(float(last_close * np.exp(cumsum)), 2)

    return {
        "ticker":      ticker,
        "task":        f"task1_3_k{k}",
        "market":      "NASDAQ",
        "last_date":   last_date,
        "last_close":  round(last_close, 2),
        "predictions": predictions,
        "unit":        "USD",
    }


@router.get("/tickers")
async def nasdaq_tickers():
    return {"tickers": NASDAQ_TICKERS, "count": len(NASDAQ_TICKERS)}


@router.get("/consecutive/{ticker}")
async def predict_nasdaq_consecutive(ticker: str, k: int = 3):
    """
    Task 1.3: k consecutive daily price forecasts for a NASDAQ ticker.
    k=3 → predictions.day_1, day_2, day_3
    k=7 → predictions.day_1 … day_7
    """
    if k not in (3, 7):
        raise HTTPException(400, "k must be 3 or 7")
    try:
        return _run_inference_consecutive(ticker.upper(), k)
    except ValueError as e:
        raise HTTPException(404, str(e))
    except Exception as e:
        raise HTTPException(500, str(e))


@router.get("/{ticker}")
async def predict_nasdaq(ticker: str, k: int = 1):
    """
    Run Task 1 LSTM inference for a NASDAQ ticker.
    k=1 → next-day price (task1_1)
    k=3 → 3rd-day price (task1_2/k3)
    k=7 → 7th-day price (task1_2/k7)
    """
    if k not in (1, 3, 7):
        raise HTTPException(400, "k must be 1, 3, or 7")
    try:
        return _run_inference(ticker.upper(), k)
    except ValueError as e:
        raise HTTPException(404, str(e))
    except Exception as e:
        raise HTTPException(500, str(e))


@router.get("/metrics/{ticker}")
async def nasdaq_metrics(ticker: str):
    """
    Return model performance metrics for a NASDAQ ticker.
    Reads metadata.json (MAE/RMSE/MAPE) and enriches with a naive
    persistence baseline computed from the last 60 local CSV rows.
    """
    import json
    ticker = ticker.upper()
    meta_path = TASK1_DIR / ticker / "metadata.json"
    if not meta_path.exists():
        raise HTTPException(404, f"No metadata for {ticker}")
    try:
        with open(meta_path) as f:
            meta = json.load(f)
    except Exception as e:
        raise HTTPException(500, f"Could not read metadata: {e}")

    # Enrich with naive baseline (persistence: tomorrow = today)
    try:
        df = _load_nasdaq_csv(ticker)
        closes = df["Close"].astype(float).values[-61:]
        if len(closes) >= 2:
            diffs          = np.diff(closes)
            naive_mae      = float(np.mean(np.abs(diffs)))
            naive_rmse     = float(np.sqrt(np.mean(diffs ** 2)))
            meta["naive_mae"]        = round(naive_mae,  4)
            meta["naive_rmse"]       = round(naive_rmse, 4)
            if naive_mae > 0:
                meta["skill_score_mae"] = round(1 - meta["test_mae"] / naive_mae, 4)
    except Exception:
        pass

    return meta
