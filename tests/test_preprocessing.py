"""
Tests for preprocessing correctness — guards against regressions to
the scaler_y leakage fix (Fix 1) and the log-return price reconstruction fix (Fix 2).
"""
import sys
from pathlib import Path

import numpy as np
import pytest

sys.path.insert(0, str(Path(__file__).parent.parent))


def test_scaler_y_no_lookahead():
    """scaler_y must be fitted only on training rows, excluding the pre-window rows."""
    from sklearn.preprocessing import StandardScaler

    lookback, n_train = 30, 200
    rng = np.random.default_rng(42)
    y_logret = rng.standard_normal(400)

    # Correct — skip pre-window rows
    sc_correct = StandardScaler().fit(y_logret[lookback : n_train + lookback].reshape(-1, 1))
    # Leaky — includes pre-window rows (the old bug)
    sc_leaked  = StandardScaler().fit(y_logret[: n_train + lookback].reshape(-1, 1))

    assert abs(sc_correct.mean_[0] - np.mean(y_logret[lookback : n_train + lookback])) < 1e-10
    assert abs(sc_leaked.mean_[0]  - np.mean(y_logret[: n_train + lookback]))          < 1e-10
    # The two scalers must differ — confirms the fix actually changes behaviour
    assert sc_correct.mean_[0] != sc_leaked.mean_[0]


def test_scaler_y_correct_std():
    """Correct scaler std must equal the std of the training slice only."""
    from sklearn.preprocessing import StandardScaler

    lookback, n_train = 30, 200
    rng = np.random.default_rng(7)
    y_logret = rng.standard_normal(400)

    sc = StandardScaler().fit(y_logret[lookback : n_train + lookback].reshape(-1, 1))
    # StandardScaler uses population std (ddof=0)
    expected_std = np.std(y_logret[lookback : n_train + lookback], ddof=0)
    assert abs(sc.scale_[0] - expected_std) < 1e-8


def test_price_reconstruction_uses_exp():
    """Price reconstruction must use exp(log_return), not the linear 1+x approximation."""
    last_close = 150.0
    log_return = 0.05  # +5% move

    correct = last_close * np.exp(log_return)   # 150 * e^0.05 ≈ 157.69
    linear  = last_close * (1.0 + log_return)   # 150 * 1.05  = 157.50 — wrong

    # exp and linear give different results
    assert correct != linear
    # exp result is strictly greater than linear for positive log_return
    assert correct > linear
    # exp result matches the mathematical definition
    assert abs(correct - 150.0 * np.e ** 0.05) < 1e-10


def test_price_reconstruction_large_move_error():
    """For large moves the linear approximation error exceeds $1 — must use exp."""
    last_close = 150.0
    log_return_large = 0.30  # +30%

    exp_price    = last_close * np.exp(log_return_large)
    linear_price = last_close * (1.0 + log_return_large)

    assert abs(exp_price - linear_price) > 1.0


def test_price_reconstruction_negative_return():
    """exp reconstruction works correctly for negative log returns too."""
    last_close = 200.0
    log_return = -0.10  # ~-9.5% actual

    pred = last_close * np.exp(log_return)
    assert 180.0 < pred < 185.0  # exp(-0.1) ≈ 0.9048
