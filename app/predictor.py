"""
predictor.py — Shared inference logic for FastAPI and Streamlit.

All model loading, feature engineering, and prediction reconstruction
lives here so the notebook, API, and UI all share a single source of truth.
"""

import json
import pathlib

import joblib
import numpy as np
import pandas as pd
import yfinance as yf

# ── Paths ─────────────────────────────────────────────────────────────────────
ROOT_DIR       = pathlib.Path(__file__).parent.parent   # project root
MODEL_BASE_DIR = ROOT_DIR / 'models'


def _task_dir(task: str) -> pathlib.Path:
    return MODEL_BASE_DIR / f'task{task.replace(".", "_")}'


def _stem(ticker: str, k: int = None) -> str:
    return f'{ticker}_k{k}' if k is not None else ticker


# ── Feature Engineering ───────────────────────────────────────────────────────
def add_technical_indicators(df: pd.DataFrame) -> pd.DataFrame:
    df    = df.copy()
    close = df['Close'];  high = df['High'];  low = df['Low']
    adj   = df['Adjusted Close'];  prev = close.shift(1)

    sma20 = close.rolling(20).mean()
    sma50 = close.rolling(50).mean()
    ema12 = close.ewm(span=12, adjust=False).mean()
    ema26 = close.ewm(span=26, adjust=False).mean()
    macd  = ema12 - ema26
    std20 = close.rolling(20).std()
    bb_upper = sma20 + 2 * std20
    bb_lower = sma20 - 2 * std20
    delta = close.diff()
    gain  = delta.clip(lower=0).rolling(14).mean()
    loss  = (-delta.clip(upper=0)).rolling(14).mean()
    rsi   = 100 - (100 / (1 + gain / (loss + 1e-9)))
    tr    = pd.concat([high - low,
                       (high - prev).abs(),
                       (low  - prev).abs()], axis=1).max(axis=1)
    atr   = tr.rolling(14).mean()
    df['Log_Return']      = np.log(close / prev).clip(-0.5, 0.5)
    df['Open_ratio']      = df['Open'] / prev - 1
    df['High_ratio']      = high       / prev - 1
    df['Low_ratio']       = low        / prev - 1
    df['Close_ratio']     = close      / prev - 1
    df['AdjClose_ratio']  = adj        / adj.shift(1) - 1
    df['Volume_norm']     = df['Volume'] / (df['Volume'].rolling(20).mean() + 1e-9)
    df['ROC_5']           = close / close.shift(5)  - 1
    df['ROC_10']          = close / close.shift(10) - 1
    df['ROC_20']          = close / close.shift(20) - 1
    df['SMA20_ratio']     = sma20 / close - 1
    df['SMA50_ratio']     = sma50 / close - 1
    df['EMA12_ratio']     = ema12 / close - 1
    df['EMA26_ratio']     = ema26 / close - 1
    df['MACD_norm']       = macd / (close + 1e-9)
    df['MACD_Signal_norm']= macd.ewm(span=9, adjust=False).mean() / (close + 1e-9)
    df['BB_width']        = (bb_upper - bb_lower) / (close + 1e-9)
    df['BB_pos']          = (close - bb_lower) / (bb_upper - bb_lower + 1e-9)
    df['ATR_norm']        = atr / (close + 1e-9)
    df['RSI_14_norm']     = rsi / 100.0
    df.dropna(inplace=True)
    df.reset_index(drop=True, inplace=True)
    return df


# ── Artifact Loading ──────────────────────────────────────────────────────────
def load_model_artifacts(ticker: str, task: str = '1.1', k: int = None):
    """Return (model, scaler, metadata) for a ticker + task."""
    import tensorflow as tf
    d      = _task_dir(task)
    s      = _stem(ticker, k)
    model  = tf.keras.models.load_model(d / f'{s}.keras')
    scaler = joblib.load(d / 'scalers' / f'{s}.joblib')
    with open(d / 'metadata' / f'{s}.json') as f:
        meta = json.load(f)
    return model, scaler, meta


# ── Recent Data Fetch ─────────────────────────────────────────────────────────
def fetch_recent(ticker: str, window: int = 120) -> pd.DataFrame:
    """
    Fetch the last `window` trading days from yfinance.
    Returns a DataFrame with the same columns as the NASDAQ CSVs.
    """
    raw = yf.download(ticker, period=f'{window * 2}d',
                      auto_adjust=False, progress=False)
    if raw.empty:
        raise ValueError(f"yfinance returned no data for {ticker}.")
    if isinstance(raw.columns, pd.MultiIndex):
        raw.columns = raw.columns.get_level_values(0)
    df = pd.DataFrame({
        'Date'          : pd.to_datetime(raw.index).tz_localize(None),
        'Open'          : raw['Open'].values,
        'High'          : raw['High'].values,
        'Low'           : raw['Low'].values,
        'Close'         : raw['Close'].values,
        'Adjusted Close': raw['Adj Close'].values,
        'Volume'        : raw['Volume'].values,
    })
    return df.tail(window).reset_index(drop=True)


# ── Prediction ────────────────────────────────────────────────────────────────
def predict_next_day(ticker: str, task: str = '1.1') -> dict:
    """
    Fetch recent data from yfinance and return next-day price prediction.
    Safe to call from FastAPI or Streamlit.
    """
    model, scaler, meta = load_model_artifacts(ticker, task)
    w    = meta['window_size']
    cols = meta['features']
    rm   = meta['ret_mean']
    rs   = meta['ret_std']
    ci   = meta.get('ci_half_90', 0.0)

    recent_df = fetch_recent(ticker, window=w + 60)  # extra rows for indicator warmup
    df_feat   = add_technical_indicators(recent_df)

    if len(df_feat) < w:
        raise ValueError(f"Need >= {w} rows; got {len(df_feat)} after feature engineering.")

    X          = scaler.transform(df_feat[cols].values[-w:])[np.newaxis]
    pred_ret   = float(model.predict(X, verbose=0).flatten()[0]) * rs + rm
    last_close = float(df_feat['Close'].iloc[-1])
    pred_price = last_close * np.exp(pred_ret)
    as_of      = str(df_feat['Date'].iloc[-1].date())

    return {
        'ticker'               : ticker,
        'task'                 : task,
        'as_of_date'           : as_of,
        'last_close'           : round(last_close, 2),
        'predicted_price'      : round(pred_price, 2),
        'predicted_return_pct' : round(pred_ret * 100, 4),
        'price_low_90ci'       : round(last_close * np.exp(pred_ret - ci), 2),
        'price_high_90ci'      : round(last_close * np.exp(pred_ret + ci), 2),
        'direction'            : 'UP' if pred_ret > 0 else 'DOWN',
    }


def list_available_tickers(task: str = '1.1') -> list[str]:
    """Return all tickers that have a saved model for this task."""
    d = _task_dir(task)
    return sorted(p.stem for p in d.glob('*.keras'))
