# tests/test_gaps.py
"""Tests for market_pipeline.gaps"""

import numpy as np
import pandas as pd
import pytest

from market_pipeline.gaps import (
    FillReport,
    GapInfo,
    detect_gaps,
    fill_gaps,
    forward_fill_prices,
    reindex_to_calendar,
)


class TestDetectGaps:

    def test_no_gaps_in_consecutive_dates(self, trading_dates):
        gaps = detect_gaps(trading_dates, threshold=5)
        # bdate_range has no gaps > weekend by construction
        assert gaps == []

    def test_detects_known_gap(self):
        d1 = pd.bdate_range("2024-01-02", periods=10)
        d2 = pd.bdate_range("2024-02-15", periods=10)  # ~45 day gap
        dates = pd.DatetimeIndex(list(d1) + list(d2))
        gaps = detect_gaps(dates, threshold=5)
        assert len(gaps) == 1
        assert gaps[0].calendar_days > 5

    def test_returns_gap_info_objects(self, trading_dates):
        d1 = pd.bdate_range("2024-01-02", periods=10)
        d2 = pd.bdate_range("2024-03-01", periods=10)
        dates = pd.DatetimeIndex(list(d1) + list(d2))
        gaps = detect_gaps(dates)
        assert all(isinstance(g, GapInfo) for g in gaps)

    def test_raises_on_fewer_than_two_dates(self):
        with pytest.raises(ValueError, match="at least 2"):
            detect_gaps(pd.DatetimeIndex(["2024-01-02"]))

    def test_gap_info_str(self, trading_dates):
        d1 = pd.bdate_range("2024-01-02", periods=5)
        d2 = pd.bdate_range("2024-03-01", periods=5)
        dates = pd.DatetimeIndex(list(d1) + list(d2))
        gaps = detect_gaps(dates)
        assert "→" in str(gaps[0])


class TestReindexToCalendar:

    def test_expands_to_full_business_day_range(self, clean_ohlcv):
        reindexed, missing = reindex_to_calendar(clean_ohlcv)
        full_range = pd.bdate_range(clean_ohlcv.index[0], clean_ohlcv.index[-1])
        assert len(reindexed) == len(full_range)

    def test_original_rows_preserved(self, clean_ohlcv):
        reindexed, _ = reindex_to_calendar(clean_ohlcv)
        for date in clean_ohlcv.index:
            assert abs(
                reindexed.loc[date, "Close"] - clean_ohlcv.loc[date, "Close"]
            ) < 1e-9

    def test_raises_on_empty_dataframe(self):
        with pytest.raises(ValueError, match="empty"):
            reindex_to_calendar(pd.DataFrame())


class TestForwardFillPrices:

    def test_fills_price_nans(self, clean_ohlcv):
        df = clean_ohlcv.copy()
        df.iloc[5, df.columns.get_loc("Close")] = float("nan")
        filled, report = forward_fill_prices(df)
        assert filled["Close"].isna().sum() == 0

    def test_volume_zero_filled_not_forward_filled(self, clean_ohlcv):
        df = clean_ohlcv.copy()
        df.iloc[5, df.columns.get_loc("Volume")] = float("nan")
        filled, _ = forward_fill_prices(df)
        assert filled.iloc[5]["Volume"] == 0.0

    def test_fill_report_counts_correctly(self, clean_ohlcv):
        df = clean_ohlcv.copy()
        df.iloc[5, df.columns.get_loc("Close")]  = float("nan")
        df.iloc[10, df.columns.get_loc("Close")] = float("nan")
        _, report = forward_fill_prices(df)
        assert report.filled_counts.get("Close", 0) == 2

    def test_fill_report_total_cells_correct(self, clean_ohlcv):
        df = clean_ohlcv.copy()
        df.iloc[5, df.columns.get_loc("Close")]  = float("nan")
        df.iloc[5, df.columns.get_loc("Volume")] = float("nan")
        _, report = forward_fill_prices(df)
        assert report.total_filled_cells == 2

    def test_no_nans_produces_empty_report(self, clean_ohlcv):
        _, report = forward_fill_prices(clean_ohlcv)
        assert report.total_filled_cells == 0
        assert report.n_rows_filled == 0

    def test_raises_on_empty_dataframe(self):
        with pytest.raises(ValueError, match="empty"):
            forward_fill_prices(pd.DataFrame())


class TestFillGaps:

    def test_returns_three_element_tuple(self, clean_ohlcv):
        result = fill_gaps(clean_ohlcv)
        assert len(result) == 3

    def test_no_nans_in_close_after_fill(self, clean_ohlcv):
        filled, _, _ = fill_gaps(clean_ohlcv)
        assert filled["Close"].isna().sum() == 0

    def test_volume_non_negative_after_fill(self, clean_ohlcv):
        filled, _, _ = fill_gaps(clean_ohlcv)
        assert (filled["Volume"] >= 0).all()