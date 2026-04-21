"""
/api/v1/signals endpoints — Task 3 buy/sell signal probabilities.
"""
import sys, json
from pathlib import Path

from fastapi import APIRouter, HTTPException
import numpy as np
import pandas as pd

ROOT = Path(__file__).parent.parent.parent
sys.path.insert(0, str(ROOT))

router = APIRouter(prefix="/signals", tags=["signals"])

_MANIFEST_PATH  = ROOT / "models" / "task3" / "task3_manifest.json"
_BUY_METRICS_PATH  = ROOT / "models" / "task3" / "task3_buy_metrics.csv"
_SELL_METRICS_PATH = ROOT / "models" / "task3" / "task3_sell_metrics.csv"

_manifest: dict | None = None

# ── Load model-performance metrics once at import time ────────────────────────
def _load_metrics() -> tuple[dict, dict]:
    """Return (buy_metrics, sell_metrics) keyed by ticker."""
    def _read(path: Path) -> dict:
        df = pd.read_csv(path, index_col=0)
        df.columns = [c.strip() for c in df.columns]
        return df.to_dict(orient="index")   # { "ACB": {AUC: 0.79, ...}, ... }

    return _read(_BUY_METRICS_PATH), _read(_SELL_METRICS_PATH)

_buy_metrics, _sell_metrics = _load_metrics()


def _load_manifest() -> dict:
    global _manifest
    if _manifest is None:
        with open(_MANIFEST_PATH) as f:
            _manifest = json.load(f)
    return _manifest


def _get_task3_signal(ticker: str) -> dict:
    """
    Run Task 3 inference for one ticker.
    Returns {buy_prob, sell_prob, threshold, signal, ticker}.
    """
    import pickle
    import tensorflow as tf

    manifest = _load_manifest()
    models   = manifest.get("models", manifest)   # support both flat and nested manifest
    if ticker not in models:
        raise ValueError(f"No Task 3 model for {ticker}")

    entry    = models[ticker]
    buy_cfg  = entry["task3_buy"]
    sell_cfg = entry["task3_sell"]

    # Load models (relative to ROOT)
    models_dir = ROOT / "models" / "task3"
    buy_model  = tf.keras.models.load_model(models_dir / buy_cfg["model"],  compile=False)
    sell_model = tf.keras.models.load_model(models_dir / sell_cfg["model"], compile=False)

    # Load scaler (shared with Task 2)
    scaler_path = ROOT / "models" / "task2" / f"scaler_X_{ticker}.pkl"
    with open(scaler_path, "rb") as f:
        scaler = pickle.load(f)

    # Load feature-engineered data
    sys.path.insert(0, str(ROOT))
    from inference import _load_raw, _add_vn_features

    df = _load_raw(ticker)
    df = _add_vn_features(df)
    df = df.dropna()

    lookback  = manifest.get("lookback", 30)
    feat_cols = manifest.get("feature_cols", [
        c for c in df.select_dtypes(include=[np.number]).columns if c.lower() != "close"
    ])

    X = df[feat_cols].values[-lookback:]
    X_scaled = scaler.transform(X)
    X_input = X_scaled.reshape(1, lookback, -1)

    buy_prob  = float(buy_model.predict(X_input, verbose=0)[0][0])
    sell_prob = float(sell_model.predict(X_input, verbose=0)[0][0])

    threshold = buy_cfg.get("threshold", 0.5)
    if buy_prob >= threshold:
        signal = "BUY"
    elif sell_prob >= sell_cfg.get("threshold", 0.5):
        signal = "SELL"
    else:
        signal = "HOLD"

    # Attach stored model-performance metrics (read from CSVs, never None)
    bm = _buy_metrics.get(ticker, {})
    sm = _sell_metrics.get(ticker, {})

    return {
        "ticker":          ticker,
        "buy_prob":        round(buy_prob, 4),
        "sell_prob":       round(sell_prob, 4),
        "threshold":       threshold,
        "signal":          signal,
        # BUY model performance
        "buy_auc":         round(float(bm.get("AUC", 0)),       3),
        "buy_f1":          round(float(bm.get("F1", 0)),        3),
        "buy_precision":   round(float(bm.get("Precision", 0)), 3),
        "buy_recall":      round(float(bm.get("Recall", 0)),    3),
        # SELL model performance
        "sell_auc":        round(float(sm.get("AUC", 0)),       3),
        "sell_f1":         round(float(sm.get("F1", 0)),        3),
        "sell_precision":  round(float(sm.get("Precision", 0)), 3),
        "sell_recall":     round(float(sm.get("Recall", 0)),    3),
    }


@router.get("/{ticker}")
async def get_signal(ticker: str):
    """Return buy/sell signal probabilities for one ticker."""
    ticker = ticker.upper()
    try:
        result = _get_task3_signal(ticker)
    except ValueError as e:
        raise HTTPException(404, str(e))
    except Exception as e:
        raise HTTPException(500, str(e))
    return result


@router.get("/all/latest")
async def get_all_signals():
    """Return signals for all tickers that have Task 3 models."""
    manifest = _load_manifest()
    tickers  = list(manifest.get("models", manifest).keys())
    results  = []
    for ticker in tickers:
        try:
            results.append(_get_task3_signal(ticker))
        except Exception as e:
            results.append({"ticker": ticker, "error": str(e)})
    return {"count": len(results), "signals": results}
