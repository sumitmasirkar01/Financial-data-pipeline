# tests/test_returns.py
"""Tests for market_pipeline.returns"""

import numpy as np
import pandas as pd
import pytest

from market_pipeline.returns import (
    ReturnStats,
    VaRResult,
    calculate_var,
    compute_max_drawdown,
    compute_return_stats,
    compute_returns,
)


class TestComputeReturns:

    def test_adds_required_columns(self, clean_ohlcv):
        result = compute_returns(clean_ohlcv)
        for col in ("simple_return", "log_return", "realized_vol_20d", "realized_vol_60d"):
            assert col in result.columns

    def test_original_columns_preserved(self, clean_ohlcv):
        result = compute_returns(clean_ohlcv)
        for col in ("Open", "High", "Low", "Close", "Volume"):
            assert col in result.columns

    def test_log_return_first_row_is_nan(self, clean_ohlcv):
        result = compute_returns(clean_ohlcv)
        assert pd.isna(result["log_return"].iloc[0])

    def test_simple_and_log_return_close_for_small_moves(self, clean_ohlcv):
        result = compute_returns(clean_ohlcv)
        sr = result["simple_return"].dropna()
        lr = result["log_return"].dropna()
        # For small returns, log ≈ simple; correlation should be near 1
        assert sr.corr(lr) > 0.999

    def test_raises_on_missing_close(self, clean_ohlcv):
        with pytest.raises(ValueError, match="'Close'"):
            compute_returns(clean_ohlcv.drop(columns=["Close"]))

    def test_raises_on_single_row(self, clean_ohlcv):
        with pytest.raises(ValueError, match="at least 2"):
            compute_returns(clean_ohlcv.iloc[[0]])


class TestComputeMaxDrawdown:

    def test_flat_returns_zero_drawdown(self):
        flat = pd.Series([0.0] * 50)
        assert compute_max_drawdown(flat) == 0.0

    def test_always_positive(self, log_returns):
        dd = compute_max_drawdown(log_returns)
        assert dd >= 0.0

    def test_known_drawdown(self):
        # Cumulative path: 1.0 → 0.80 → 0.60 → 0.80 → 1.0
        # Peak before trough is 0.80, trough is 0.60
        # Drawdown = (0.60 - 0.80) / 0.80 = 0.25
        returns = pd.Series([-0.20, -0.25, 1/3, 0.25])
        dd = compute_max_drawdown(returns)
        assert abs(dd - 0.25) < 1e-9

    def test_empty_series_returns_zero(self):
        assert compute_max_drawdown(pd.Series([], dtype=float)) == 0.0


class TestComputeReturnStats:

    def test_returns_return_stats_object(self, log_returns, clean_ohlcv):
        simple = clean_ohlcv["Close"].pct_change().dropna()
        result = compute_return_stats(log_returns, simple)
        assert isinstance(result, ReturnStats)

    def test_sharpe_zero_when_vol_zero(self):
        # Zero returns → zero vol → Sharpe guard returns 0.0
        flat = pd.Series([0.0] * 100)
        simple = pd.Series([0.0] * 100)
        stats = compute_return_stats(flat, simple)
        assert stats.sharpe == 0.0

    def test_n_obs_correct(self, log_returns, clean_ohlcv):
        simple = clean_ohlcv["Close"].pct_change().dropna()
        stats = compute_return_stats(log_returns, simple)
        assert stats.n_obs == len(log_returns)

    def test_max_drawdown_positive(self, log_returns, clean_ohlcv):
        simple = clean_ohlcv["Close"].pct_change().dropna()
        stats = compute_return_stats(log_returns, simple)
        assert stats.max_drawdown >= 0.0

    def test_raises_when_too_short(self):
        tiny = pd.Series([0.001])
        with pytest.raises(ValueError, match="at least 2"):
            compute_return_stats(tiny, tiny)


class TestCalculateVar:

    def test_returns_var_result(self, log_returns):
        result = calculate_var(log_returns)
        assert isinstance(result, VaRResult)

    def test_historical_var_is_positive(self, log_returns):
        result = calculate_var(log_returns)
        assert result.historical_var_pct > 0.0

    def test_parametric_var_is_positive(self, log_returns):
        result = calculate_var(log_returns)
        assert result.parametric_var_pct > 0.0

    def test_cash_var_equals_pct_times_principal(self, log_returns):
        result = calculate_var(log_returns, principal=200_000)
        assert abs(
            result.historical_var_cash - result.historical_var_pct * 200_000
        ) < 1e-6

    def test_historical_var_ge_parametric_for_fat_tails(self):
        """Left-tail outlier: historical VaR should exceed parametric."""
        # We build a highly deterministic "fat tail" distribution.
        # 98 days of pure silence (0% return), and 2 days of massive crashes (-20%).
        returns = [0.0] * 98 + [-0.20, -0.20]
        r = pd.Series(returns)
        
        # At 99% confidence, Historical VaR will look right at the 1st percentile
        # and see the massive -20% crashes. 
        # Parametric VaR will look at the low overall standard deviation and underestimate the risk.
        result = calculate_var(r, confidence_level=0.99)
        
        assert result.historical_var_pct >= result.parametric_var_pct

    def test_raises_on_bad_confidence_level(self, log_returns):
        with pytest.raises(ValueError, match="confidence_level"):
            calculate_var(log_returns, confidence_level=1.5)

    def test_higher_confidence_gives_larger_var(self, log_returns):
        var_95 = calculate_var(log_returns, confidence_level=0.95)
        var_99 = calculate_var(log_returns, confidence_level=0.99)
        assert var_99.historical_var_pct >= var_95.historical_var_pct