"""
OHLCV data validation — shared by predict_nasdaq.py and inference.py.
Cleans and sanity-checks a price DataFrame before feature engineering.
"""
import logging
import numpy as np
import pandas as pd

logger = logging.getLogger("ohlcv_validation")


def validate_ohlcv(df: pd.DataFrame, ticker: str = "") -> pd.DataFrame:
    """
    Clean and validate an OHLCV DataFrame. Returns a sanitized copy.

    Checks performed:
      1. Drop duplicate dates (keep last — most recent data wins).
      2. Drop rows with zero or negative prices.
      3. Fix High < Low violations via forward-fill.
      4. Warn if >5% of rows have zero volume.
      5. Warn if the latest data is older than 14 calendar days.
    """
    df = df.copy()
    date_col = "date" if "date" in df.columns else "Date"

    # 1. Duplicate dates
    before = len(df)
    df = df.drop_duplicates(subset=[date_col], keep="last")
    if len(df) < before:
        logger.warning("%s: dropped %d duplicate date rows", ticker, before - len(df))

    # 2. Zero / negative prices
    price_cols = [c for c in df.columns if c.lower() in ("open", "high", "low", "close")]
    for col in price_cols:
        df[col] = pd.to_numeric(df[col], errors="coerce")
    bad_price = (df[price_cols] <= 0).any(axis=1)
    if bad_price.sum():
        logger.warning("%s: dropping %d rows with zero/negative prices", ticker, bad_price.sum())
        df = df[~bad_price]

    # 3. OHLC ordering violation: High < Low
    h_col = next((c for c in df.columns if c.lower() == "high"), None)
    l_col = next((c for c in df.columns if c.lower() == "low"),  None)
    if h_col and l_col:
        bad_hl = df[h_col] < df[l_col]
        if bad_hl.sum():
            logger.warning("%s: %d rows with High < Low — forward-filling", ticker, bad_hl.sum())
            df.loc[bad_hl, [h_col, l_col]] = np.nan
            df[[h_col, l_col]] = df[[h_col, l_col]].ffill(limit=2)

    # 4. Zero volume (warning only — some tickers have legitimate 0-volume days)
    vol_col = next((c for c in df.columns if c.lower() == "volume"), None)
    if vol_col:
        zero_vol = (df[vol_col] == 0).sum()
        if zero_vol > len(df) * 0.05:
            logger.warning(
                "%s: %.1f%% rows have zero volume", ticker, 100 * zero_vol / len(df)
            )

    # 5. Stale data check
    try:
        last_date = pd.to_datetime(df[date_col].iloc[-1])
        days_old  = (pd.Timestamp.now() - last_date).days
        if days_old > 14:
            logger.warning(
                "%s: latest data is %d days old (%s)", ticker, days_old, last_date.date()
            )
    except Exception:
        pass

    return df.reset_index(drop=True)
