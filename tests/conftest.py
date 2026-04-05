# tests/conftest.py
"""
Shared pytest fixtures used across all test modules.

All fixtures use synthetic data — no network calls, no yfinance.
Tests run offline and deterministically.
"""

import numpy as np
import pandas as pd
import pytest

from market_pipeline.adjustments import DividendEvent, SplitEvent
from market_pipeline.validator import ValidationResult


# ---------------------------------------------------------------------------
# Date ranges
# ---------------------------------------------------------------------------

@pytest.fixture
def trading_dates() -> pd.DatetimeIndex:
    """100 business days starting 2024-01-02."""
    return pd.bdate_range("2024-01-02", periods=100)


# ---------------------------------------------------------------------------
# Clean OHLCV DataFrames
# ---------------------------------------------------------------------------

@pytest.fixture
def clean_ohlcv(trading_dates) -> pd.DataFrame:
    """
    Structurally valid OHLCV DataFrame.
    Close drifts upward with small random noise.
    High >= Close >= Low is always satisfied.
    """
    rng = np.random.default_rng(42)
    n = len(trading_dates)

    close = 150.0 + np.cumsum(rng.normal(0.1, 1.5, n))
    close = np.maximum(close, 1.0)          # never go to zero

    spread = rng.uniform(0.5, 2.0, n)
    high   = close + spread
    low    = close - spread
    open_  = low + rng.uniform(0, 1, n) * spread * 2

    return pd.DataFrame({
        "Open":   open_,
        "High":   high,
        "Low":    low,
        "Close":  close,
        "Volume": rng.integers(5_000_000, 50_000_000, n),
    }, index=trading_dates)


@pytest.fixture
def dirty_ohlcv(clean_ohlcv) -> pd.DataFrame:
    """
    OHLCV DataFrame with known injected violations:
      - Row 0: High < Low  (swapped)
      - Row 5: negative volume
      - Row 10: NaN close
      - Row 15: zero close
    """
    df = clean_ohlcv.copy()
    df.iloc[0, df.columns.get_loc("High")] = df.iloc[0]["Low"] - 1.0
    df.iloc[5, df.columns.get_loc("Volume")] = -500
    df.iloc[10, df.columns.get_loc("Close")] = float("nan")
    df.iloc[15, df.columns.get_loc("Close")] = 0.0
    return df


# ---------------------------------------------------------------------------
# Return series
# ---------------------------------------------------------------------------

@pytest.fixture
def log_returns(clean_ohlcv) -> pd.Series:
    """Log returns derived from clean_ohlcv."""
    import numpy as np
    close = clean_ohlcv["Close"]
    return np.log(close / close.shift(1)).dropna()


@pytest.fixture
def returns_with_outlier(log_returns) -> pd.Series:
    """Log returns with one injected extreme value at position 50."""
    r = log_returns.copy()
    r.iloc[50] = 0.45       # +45% in one day — extreme outlier
    return r


# ---------------------------------------------------------------------------
# Corporate action events
# ---------------------------------------------------------------------------

@pytest.fixture
def split_4_for_1(trading_dates) -> SplitEvent:
    """4:1 split on the 50th trading day."""
    return SplitEvent(date=str(trading_dates[50].date()), ratio=4)


@pytest.fixture
def dividend_150(trading_dates) -> DividendEvent:
    """$1.50 dividend with ex-date on trading day 30."""
    return DividendEvent(ex_date=str(trading_dates[30].date()), amount=1.50)


# ---------------------------------------------------------------------------
# Minimal pipeline result dict (for invariant checker)
# ---------------------------------------------------------------------------

@pytest.fixture
def good_pipeline_result(clean_ohlcv):
    """Minimal pipeline result dict that should pass all invariants."""
    import numpy as np
    df = clean_ohlcv.copy()
    df["is_outlier"] = False
    return {
        "clean_data":    df,
        "quality_score": 0.97,
    }