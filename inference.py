"""
inference.py — Task 2 prediction helper
Loaded by Task 5 (REST API, SaaS, automation pipeline).

Usage:
    from inference import predict_price, predict_all_tickers, TICKERS

    # Single ticker, single subtask
    result = predict_price('FPT', task='task2_1')
    # {'ticker': 'FPT', 'task': 'task2_1', 'last_close': 77500.0,
    #  'predictions': {'day_1': 77616.46}, 'unit': 'VND'}

    # All tickers, next-day
    results = predict_all_tickers(task='task2_1')
"""

import pickle, json, warnings
import numpy as np
import pandas as pd
import tensorflow as tf
from pathlib import Path

warnings.filterwarnings('ignore')
tf.get_logger().setLevel('ERROR')

# ── Paths ──────────────────────────────────────────────────────────────────────
_ROOT      = Path(__file__).parent
_ART_DIR   = _ROOT / 'models' / 'task2'
_DATA_DIR  = _ROOT / 'clean-historical-data-2026'

with open(_ART_DIR / 'model_manifest.json') as f:
    _MANIFEST = json.load(f)

TICKERS      = _MANIFEST['tickers']
FEATURE_COLS = _MANIFEST['feature_cols']
LOOKBACK     = _MANIFEST['lookback']   # 30
N_FEATURES   = _MANIFEST['n_features'] # 24

VALID_TASKS  = ['task2_1', 'task2_2_k3', 'task2_2_k7', 'task2_3_k3', 'task2_3_k7']

# ── Model cache (load once per process, not per request) ───────────────────────
_model_cache  = {}
_scaler_cache = {}

def _get_model(path_str: str):
    if path_str not in _model_cache:
        _model_cache[path_str] = tf.keras.models.load_model(path_str)
    return _model_cache[path_str]

def _get_scaler(path_str: str):
    if path_str not in _scaler_cache:
        with open(path_str, 'rb') as f:
            _scaler_cache[path_str] = pickle.load(f)
    return _scaler_cache[path_str]


# ── Feature engineering (must match notebook cell ea04e6ec exactly) ───────────
def _add_vn_features(df: pd.DataFrame) -> pd.DataFrame:
    df     = df.copy().reset_index(drop=True)
    close  = df['Close']
    high   = df['High']
    low    = df['Low']
    op     = df['Open']
    volume = df['Volume']
    c_safe = close.replace(0, np.nan)

    sma10  = close.rolling(10).mean()
    sma20  = close.rolling(20).mean()
    sma50  = close.rolling(50).mean()
    ema10  = close.ewm(span=10, adjust=False).mean()
    ema20  = close.ewm(span=20, adjust=False).mean()
    df['SMA10_vs_SMA20'] = sma10 / sma20.replace(0, np.nan) - 1
    df['SMA20_vs_SMA50'] = sma20 / sma50.replace(0, np.nan) - 1
    df['EMA10_vs_EMA20'] = ema10 / ema20.replace(0, np.nan) - 1
    df['Close_vs_SMA20'] = close / sma20.replace(0, np.nan) - 1
    df['Close_vs_SMA50'] = close / sma50.replace(0, np.nan) - 1

    delta = close.diff()
    gain  = delta.clip(lower=0).rolling(14).mean()
    loss  = (-delta).clip(lower=0).rolling(14).mean()
    df['RSI_14'] = 100 - (100 / (1 + gain / loss.replace(0, np.nan)))

    ema12 = close.ewm(span=12, adjust=False).mean()
    ema26 = close.ewm(span=26, adjust=False).mean()
    macd  = ema12 - ema26
    msig  = macd.ewm(span=9, adjust=False).mean()
    df['MACD_pct']      = macd          / c_safe
    df['MACD_sig_pct']  = msig          / c_safe
    df['MACD_hist_pct'] = (macd - msig) / c_safe

    bb_mid = close.rolling(20).mean()
    bb_std = close.rolling(20).std()
    bb_rng = (4 * bb_std).replace(0, np.nan)
    df['BB_width'] = bb_rng / bb_mid.replace(0, np.nan)
    df['BB_pct']   = (close - (bb_mid - 2 * bb_std)) / bb_rng

    tr = pd.concat([high - low,
                    (high - close.shift(1)).abs(),
                    (low  - close.shift(1)).abs()], axis=1).max(axis=1)
    df['ATR_pct'] = tr.rolling(14).mean() / c_safe

    log_ret = np.log(close / close.shift(1))
    df['Volatility_10'] = log_ret.rolling(10).std()
    df['Volatility_20'] = log_ret.rolling(20).std()
    df['Return_1d']     = log_ret
    df['Return_5d']     = np.log(close / close.shift(5))
    df['Return_10d']    = np.log(close / close.shift(10))
    df['Return_20d']    = np.log(close / close.shift(20))

    df['Volume_ratio'] = volume / volume.rolling(20).mean().replace(0, np.nan)
    df['High_Low_pct'] = (high - low) / c_safe

    df['VN_limit_prox'] = log_ret / 0.07
    df['OC_pct']        = (close - op) / c_safe
    df['Upper_shadow']  = (high - pd.concat([op, close], axis=1).max(axis=1)) / c_safe
    df['Lower_shadow']  = (pd.concat([op, close], axis=1).min(axis=1) - low)  / c_safe
    return df


def _load_raw(ticker: str) -> pd.DataFrame:
    path = _DATA_DIR / f'{ticker}_Historical.csv'
    df   = pd.read_csv(path, skiprows=[1])
    df['Date'] = pd.to_datetime(df['Date'], errors='coerce')
    df = df.dropna(subset=['Date']).sort_values('Date').reset_index(drop=True)
    for col in ['Open', 'High', 'Low', 'Close']:
        df[col] = pd.to_numeric(df[col], errors='coerce').clip(lower=1)
    df['Volume'] = pd.to_numeric(df['Volume'], errors='coerce').fillna(0)
    df[['Open','High','Low','Close']] = df[['Open','High','Low','Close']].ffill(limit=3)
    return df.dropna(subset=['Open','High','Low','Close']).reset_index(drop=True)


# ── Public API ─────────────────────────────────────────────────────────────────
def predict_price(ticker: str, task: str = 'task2_1') -> dict:
    """
    Run inference for one ticker and one task.

    Parameters
    ----------
    ticker : str   e.g. 'FPT', 'STB'
    task   : str   one of VALID_TASKS

    Returns
    -------
    dict with keys: ticker, task, last_close, predictions {day_N: float}, unit
    """
    if ticker not in TICKERS:
        raise ValueError(f"Unknown ticker '{ticker}'. Available: {TICKERS}")
    if task not in VALID_TASKS:
        raise ValueError(f"Unknown task '{task}'. Available: {VALID_TASKS}")

    entry    = _MANIFEST['models'][ticker][task]
    model    = _get_model(str(_ART_DIR / entry['model']))
    scaler_X = _get_scaler(str(_ART_DIR / entry['scaler_X']))
    scaler_y = _get_scaler(str(_ART_DIR / entry['scaler_y']))

    df_raw   = _load_raw(ticker)
    df_feat  = _add_vn_features(df_raw).dropna(subset=FEATURE_COLS).reset_index(drop=True)

    if len(df_feat) < LOOKBACK:
        raise ValueError(f"{ticker}: insufficient data ({len(df_feat)} rows, need {LOOKBACK})")

    window  = df_feat[FEATURE_COLS].values[-LOOKBACK:]
    X       = scaler_X.transform(window).reshape(1, LOOKBACK, N_FEATURES).astype('float32')
    pred_sc = model.predict(X, verbose=0)
    log_rets = scaler_y.inverse_transform(pred_sc.reshape(-1, 1)).flatten()

    last_close = float(df_feat['Close'].iloc[-1])
    last_date  = str(pd.Timestamp(df_feat['Date'].iloc[-1]).date()) if 'Date' in df_feat.columns else None

    output_size = entry['output_size']
    if output_size == 1:
        predictions = {'day_1': round(float(last_close * np.exp(log_rets[0])), 2)}
    else:
        predictions, cumsum = {}, 0.0
        for j, r in enumerate(log_rets):
            cumsum += float(r)
            predictions[f'day_{j+1}'] = round(float(last_close * np.exp(cumsum)), 2)

    return {
        'ticker'     : ticker,
        'task'       : task,
        'last_date'  : last_date,
        'last_close' : round(last_close, 2),
        'predictions': predictions,
        'unit'       : 'VND',
    }


def predict_all_tickers(task: str = 'task2_1') -> list[dict]:
    """Run predict_price for every ticker. Returns list of result dicts."""
    results, errors = [], []
    for ticker in TICKERS:
        try:
            results.append(predict_price(ticker, task=task))
        except Exception as e:
            errors.append({'ticker': ticker, 'error': str(e)})
    if errors:
        print(f"  Errors: {errors}")
    return results


# ── Quick self-test ────────────────────────────────────────────────────────────
if __name__ == '__main__':
    print("Running inference self-test ...\n")
    for task in VALID_TASKS:
        r = predict_price('FPT', task=task)
        preds = r['predictions']
        print(f"  {task:<15}  last_close={r['last_close']:>10.2f}  "
              f"predictions={preds}")
    print("\nSelf-test passed — inference.py is deployment-ready.")
