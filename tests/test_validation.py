"""
Tests for the OHLCV validation utility (api/utils/validation.py).
"""
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

sys.path.insert(0, str(Path(__file__).parent.parent))

from api.utils.validation import validate_ohlcv


def _make_df(**overrides):
    """Build a minimal valid OHLCV DataFrame, optionally overriding columns."""
    base = {
        "date":   ["2025-01-01", "2025-01-02", "2025-01-03"],
        "open":   [100.0, 101.0, 102.0],
        "high":   [105.0, 106.0, 107.0],
        "low":    [99.0,  100.0, 101.0],
        "close":  [103.0, 104.0, 105.0],
        "volume": [1000,  1100,  1200],
    }
    base.update(overrides)
    return pd.DataFrame(base)


def test_valid_df_passes_through():
    df  = _make_df()
    out = validate_ohlcv(df)
    assert len(out) == 3


def test_removes_duplicate_dates():
    df = _make_df(
        date=["2025-01-01", "2025-01-01", "2025-01-02"],
        open=[100.0, 999.0, 102.0],
        close=[103.0, 888.0, 105.0],
    )
    out = validate_ohlcv(df)
    assert len(out) == 2
    # Last duplicate (999/888) should be kept
    assert out["open"].iloc[0] == 999.0


def test_drops_zero_price_rows():
    df = _make_df(
        open=[0.0, 101.0, 102.0],
        high=[0.0, 106.0, 107.0],
        low=[0.0, 100.0, 101.0],
        close=[0.0, 104.0, 105.0],
    )
    out = validate_ohlcv(df)
    assert len(out) == 2
    assert (out["close"] > 0).all()


def test_drops_negative_price_rows():
    df = _make_df(close=[103.0, -5.0, 105.0])
    out = validate_ohlcv(df)
    assert len(out) == 2


def test_fixes_high_less_than_low():
    # Row 1 has High < Low — should be forward-filled
    df = _make_df(high=[105.0, 98.0, 107.0], low=[99.0, 100.0, 101.0])
    out = validate_ohlcv(df)
    # After forward-fill the bad row inherits previous values; no crash
    assert len(out) == 3


def test_returns_reset_index():
    df  = _make_df()
    out = validate_ohlcv(df)
    assert list(out.index) == [0, 1, 2]
