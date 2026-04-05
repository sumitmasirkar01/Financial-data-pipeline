# tests/test_outliers.py
"""Tests for market_pipeline.outliers"""

import numpy as np
import pandas as pd
import pytest

from market_pipeline.outliers import (
    OutlierResult,
    compare_methods,
    compute_mad,
    detect_outliers_mad,
    global_modified_z_scores,
)


class TestComputeMad:

    def test_known_series(self):
        s = pd.Series([1.0, 2.0, 3.0, 4.0, 5.0])
        assert abs(compute_mad(s) - 1.0) < 1e-10

    def test_constant_series_returns_zero(self):
        s = pd.Series([5.0, 5.0, 5.0, 5.0])
        assert compute_mad(s) == 0.0

    def test_single_value_returns_zero(self):
        s = pd.Series([42.0])
        assert compute_mad(s) == 0.0

    def test_resistant_to_outlier(self):
        clean = pd.Series([1.0, 2.0, 3.0, 4.0, 5.0])
        contaminated = pd.Series([1.0, 2.0, 3.0, 4.0, 100.0])
        # MAD should barely change — unlike std which inflates massively
        assert compute_mad(contaminated) / compute_mad(clean) < 2.0


class TestGlobalModifiedZScores:

    def test_zero_mean_series_has_zero_centre(self):
        s = pd.Series([-2.0, -1.0, 0.0, 1.0, 2.0])
        z = global_modified_z_scores(s)
        assert abs(z.loc[z.index[2]]) < 1e-9     # middle value should be 0

    def test_constant_series_returns_zeros(self):
        s = pd.Series([3.0, 3.0, 3.0, 3.0])
        z = global_modified_z_scores(s)
        assert (z == 0.0).all()


class TestDetectOutliersMad:

    def test_returns_outlier_result(self, log_returns):
        result = detect_outliers_mad(log_returns)
        assert isinstance(result, OutlierResult)

    def test_flags_extreme_return(self, returns_with_outlier):
        result = detect_outliers_mad(returns_with_outlier, window=30, threshold=3.5)
        assert result.n_outliers >= 1

    def test_clean_returns_have_low_outlier_rate(self, log_returns):
        result = detect_outliers_mad(log_returns)
        assert result.outlier_rate < 0.05

    def test_outlier_dates_match_is_outlier_mask(self, returns_with_outlier):
        result = detect_outliers_mad(returns_with_outlier, window=30)
        flagged_by_mask  = set(result.is_outlier.index[result.is_outlier].tolist())
        flagged_by_dates = set(result.outlier_dates)
        assert flagged_by_mask == flagged_by_dates

    def test_zero_mad_window_produces_nan_not_inf(self, trading_dates):
        """Constant window → scaled_mad=0 → z-score should be NaN, not inf."""
        constant = pd.Series(0.001, index=trading_dates)
        result = detect_outliers_mad(constant, window=10, min_periods=5)
        assert not np.isinf(result.modified_z).any()

    def test_raises_on_empty_series(self):
        with pytest.raises(ValueError, match="empty"):
            detect_outliers_mad(pd.Series([], dtype=float))

    def test_raises_when_too_short(self):
        tiny = pd.Series([0.01, 0.02, 0.03])
        with pytest.raises(ValueError, match="min_periods"):
            detect_outliers_mad(tiny, min_periods=20)

    def test_outlier_rate_property(self, returns_with_outlier):
        result = detect_outliers_mad(returns_with_outlier, window=30)
        expected = result.n_outliers / int(result.is_outlier.notna().sum())
        assert abs(result.outlier_rate - expected) < 1e-9


class TestCompareMethods:

    def test_returns_expected_keys(self, log_returns):
        out = compare_methods(log_returns)
        for key in ("z_outliers", "mad_result", "n_z", "n_mad", "n_both", "n_only_mad"):
            assert key in out

    def test_n_only_mad_non_negative(self, log_returns):
        out = compare_methods(log_returns)
        assert out["n_only_mad"] >= 0