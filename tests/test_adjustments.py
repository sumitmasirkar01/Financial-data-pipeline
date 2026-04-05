# tests/test_adjustments.py
"""Tests for market_pipeline.adjustments"""

import pytest
import pandas as pd
import numpy as np

from market_pipeline.adjustments import (
    SplitEvent,
    DividendEvent,
    adjust_for_splits,
    adjust_for_dividends,
    apply_corporate_actions,
)


class TestSplitEvent:

    def test_valid_construction(self):
        s = SplitEvent(date="2024-01-15", ratio=4)
        assert s.ratio == 4
        assert isinstance(s.date, pd.Timestamp)

    def test_raises_on_zero_ratio(self):
        with pytest.raises(ValueError, match="positive"):
            SplitEvent(date="2024-01-15", ratio=0)

    def test_raises_on_negative_ratio(self):
        with pytest.raises(ValueError, match="positive"):
            SplitEvent(date="2024-01-15", ratio=-2)


class TestDividendEvent:

    def test_valid_construction(self):
        d = DividendEvent(ex_date="2024-02-15", amount=1.50)
        assert d.amount == 1.50
        assert isinstance(d.ex_date, pd.Timestamp)

    def test_raises_on_zero_amount(self):
        with pytest.raises(ValueError, match="positive"):
            DividendEvent(ex_date="2024-02-15", amount=0)

    def test_raises_on_negative_amount(self):
        with pytest.raises(ValueError, match="positive"):
            DividendEvent(ex_date="2024-02-15", amount=-1.0)


class TestAdjustForSplits:

    def test_pre_split_prices_divided_by_ratio(self, clean_ohlcv, split_4_for_1):
        raw_pre = clean_ohlcv.iloc[0]["Close"]
        adjusted, _ = adjust_for_splits(clean_ohlcv, [split_4_for_1])
        adj_pre = adjusted.iloc[0]["Close"]
        assert abs(adj_pre - raw_pre / 4) < 1e-9

    def test_post_split_prices_unchanged(self, clean_ohlcv, split_4_for_1):
        raw_post = clean_ohlcv.iloc[-1]["Close"]
        adjusted, _ = adjust_for_splits(clean_ohlcv, [split_4_for_1])
        adj_post = adjusted.iloc[-1]["Close"]
        assert abs(adj_post - raw_post) < 1e-9

    def test_pre_split_volume_multiplied_by_ratio(self, clean_ohlcv, split_4_for_1):
        raw_vol = clean_ohlcv.iloc[0]["Volume"]
        adjusted, _ = adjust_for_splits(clean_ohlcv, [split_4_for_1])
        adj_vol = adjusted.iloc[0]["Volume"]
        assert abs(adj_vol - raw_vol * 4) < 1e-3

    def test_report_records_applied_split(self, clean_ohlcv, split_4_for_1):
        _, report = adjust_for_splits(clean_ohlcv, [split_4_for_1])
        assert report.n_splits == 1
        assert report.splits_applied[0]["ratio"] == 4

    def test_empty_splits_returns_unchanged(self, clean_ohlcv):
        adjusted, report = adjust_for_splits(clean_ohlcv, [])
        pd.testing.assert_frame_equal(adjusted, clean_ohlcv)
        assert report.n_splits == 0

    def test_splits_applied_in_chronological_order(self, clean_ohlcv, trading_dates):
        """Two splits passed out of order must produce same result as in order."""
        s1 = SplitEvent(date=str(trading_dates[30].date()), ratio=2)
        s2 = SplitEvent(date=str(trading_dates[60].date()), ratio=3)

        adj_ordered, _   = adjust_for_splits(clean_ohlcv, [s1, s2])
        adj_reversed, _  = adjust_for_splits(clean_ohlcv, [s2, s1])
        pd.testing.assert_frame_equal(adj_ordered, adj_reversed)

    def test_split_before_all_data_is_skipped(self, clean_ohlcv):
        early_split = SplitEvent(date="1990-01-01", ratio=2)
        _, report = adjust_for_splits(clean_ohlcv, [early_split])
        assert report.n_splits == 0
        assert len(report.splits_skipped) == 1


class TestAdjustForDividends:

    def test_pre_ex_date_price_multiplied_by_factor(self, clean_ohlcv, dividend_150):
        prior_close = clean_ohlcv.loc[
            clean_ohlcv.index < dividend_150.ex_date, "Close"
        ].iloc[-1]
        expected_factor = 1 - 1.50 / prior_close

        raw_price  = clean_ohlcv.iloc[0]["Close"]
        adjusted, _ = adjust_for_dividends(clean_ohlcv, [dividend_150])
        adj_price  = adjusted.iloc[0]["Close"]

        assert abs(adj_price - raw_price * expected_factor) < 1e-8

    def test_post_ex_date_price_unchanged(self, clean_ohlcv, dividend_150):
        raw_post  = clean_ohlcv.iloc[-1]["Close"]
        adjusted, _ = adjust_for_dividends(clean_ohlcv, [dividend_150])
        adj_post  = adjusted.iloc[-1]["Close"]
        assert abs(adj_post - raw_post) < 1e-9

    def test_report_records_factor(self, clean_ohlcv, dividend_150):
        _, report = adjust_for_dividends(clean_ohlcv, [dividend_150])
        assert report.n_dividends == 1
        factor = report.dividends_applied[0]["factor"]
        assert 0.0 < factor < 1.0

    def test_invalid_factor_is_skipped(self, clean_ohlcv, trading_dates):
        """Dividend larger than prior close → factor < 0 → must skip."""
        huge_div = DividendEvent(
            ex_date=str(trading_dates[30].date()),
            amount=99999.0,
        )
        adjusted, report = adjust_for_dividends(clean_ohlcv, [huge_div])
        assert report.n_dividends == 0
        assert len(report.dividends_skipped) == 1
        pd.testing.assert_frame_equal(adjusted, clean_ohlcv)

    def test_empty_dividends_returns_unchanged(self, clean_ohlcv):
        adjusted, report = adjust_for_dividends(clean_ohlcv, [])
        pd.testing.assert_frame_equal(adjusted, clean_ohlcv)
        assert report.n_dividends == 0


class TestApplyCorporateActions:

    def test_splits_applied_before_dividends(
        self, clean_ohlcv, split_4_for_1, dividend_150
    ):
        """Applying splits then dividends vs the combined function must match."""
        adj_split, _ = adjust_for_splits(clean_ohlcv, [split_4_for_1])
        adj_both_manual, _ = adjust_for_dividends(adj_split, [dividend_150])

        adj_combined, _ = apply_corporate_actions(
            clean_ohlcv,
            splits=[split_4_for_1],
            dividends=[dividend_150],
        )
        pd.testing.assert_frame_equal(adj_combined, adj_both_manual)

    def test_no_actions_returns_unchanged(self, clean_ohlcv):
        adjusted, report = apply_corporate_actions(clean_ohlcv)
        pd.testing.assert_frame_equal(adjusted, clean_ohlcv)
        assert report.n_splits == 0
        assert report.n_dividends == 0