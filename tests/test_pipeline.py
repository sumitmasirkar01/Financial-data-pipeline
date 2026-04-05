# tests/test_pipeline.py
"""
Integration tests for market_pipeline.pipeline

These tests run the full pipeline end-to-end on synthetic data.
No network calls — yfinance is never touched.
"""

import numpy as np
import pandas as pd
import pytest

from market_pipeline.adjustments import DividendEvent, SplitEvent
from market_pipeline.pipeline import PipelineResult, run_pipeline
from market_pipeline.scoring import QualityScore


class TestRunPipeline:

    def test_returns_pipeline_result(self, clean_ohlcv):
        result = run_pipeline(clean_ohlcv)
        assert isinstance(result, PipelineResult)

    def test_clean_data_is_dataframe(self, clean_ohlcv):
        result = run_pipeline(clean_ohlcv)
        assert isinstance(result.clean_data, pd.DataFrame)

    def test_quality_is_quality_score(self, clean_ohlcv):
        result = run_pipeline(clean_ohlcv)
        assert isinstance(result.quality, QualityScore)

    def test_grade_is_valid_letter(self, clean_ohlcv):
        result = run_pipeline(clean_ohlcv)
        assert result.quality.grade in {"A", "B", "C", "D", "F"}

    def test_quality_score_in_unit_interval(self, clean_ohlcv):
        result = run_pipeline(clean_ohlcv)
        assert 0.0 <= result.quality.quality_score <= 1.0

    def test_clean_data_has_return_columns(self, clean_ohlcv):
        result = run_pipeline(clean_ohlcv)
        for col in ("simple_return", "log_return"):
            assert col in result.clean_data.columns

    def test_clean_data_has_outlier_columns(self, clean_ohlcv):
        result = run_pipeline(clean_ohlcv)
        assert "is_outlier" in result.clean_data.columns
        assert "modified_z" in result.clean_data.columns

    def test_no_nan_close_after_pipeline(self, clean_ohlcv):
        result = run_pipeline(clean_ohlcv)
        assert result.clean_data["Close"].isna().sum() == 0

    def test_no_negative_volume_after_pipeline(self, clean_ohlcv):
        result = run_pipeline(clean_ohlcv)
        assert (result.clean_data["Volume"] >= 0).all()

    def test_invariant_failures_empty_on_clean_data(self, clean_ohlcv):
        result = run_pipeline(clean_ohlcv)
        assert result.invariant_failures == []

    def test_metadata_has_required_keys(self, clean_ohlcv):
        result = run_pipeline(clean_ohlcv)
        for key in ("pipeline_version", "timestamp", "outlier_window",
                    "outlier_threshold", "calendar"):
            assert key in result.metadata

    def test_metadata_timestamp_is_utc_string(self, clean_ohlcv):
        result = run_pipeline(clean_ohlcv)
        assert result.metadata["timestamp"].endswith("Z")

    def test_return_stats_populated(self, clean_ohlcv):
        result = run_pipeline(clean_ohlcv)
        assert result.return_stats is not None
        assert result.return_stats.n_obs > 0

    def test_raises_on_empty_dataframe(self):
        with pytest.raises(ValueError, match="empty"):
            run_pipeline(pd.DataFrame())


class TestRunPipelineWithCorporateActions:

    def test_split_adjustment_applied(self, clean_ohlcv, split_4_for_1):
        raw_pre_price = clean_ohlcv.iloc[0]["Close"]
        result = run_pipeline(clean_ohlcv, splits=[split_4_for_1])

        # The pre-split date should be in the clean data — price divided by 4
        pre_date = clean_ohlcv.index[0]
        adj_price = result.clean_data.loc[pre_date, "Close"]
        assert abs(adj_price - raw_pre_price / 4) < 1e-6

    def test_adjustment_report_records_split(self, clean_ohlcv, split_4_for_1):
        result = run_pipeline(clean_ohlcv, splits=[split_4_for_1])
        assert result.adjustment_report.n_splits == 1

    def test_dividend_adjustment_applied(self, clean_ohlcv, dividend_150):
        result = run_pipeline(clean_ohlcv, dividends=[dividend_150])
        assert result.adjustment_report.n_dividends == 1

    def test_no_actions_still_succeeds(self, clean_ohlcv):
        result = run_pipeline(clean_ohlcv, splits=None, dividends=None)
        assert result.adjustment_report.n_splits == 0
        assert result.adjustment_report.n_dividends == 0


class TestRunPipelineOnDirtyData:

    def test_pipeline_completes_on_dirty_data(self, dirty_ohlcv):
        """Pipeline should complete even on bad input — it cleans, not rejects."""
        result = run_pipeline(dirty_ohlcv)
        assert isinstance(result, PipelineResult)

    def test_dirty_data_scores_lower_than_clean(self, clean_ohlcv, dirty_ohlcv):
        clean_result = run_pipeline(clean_ohlcv)
        dirty_result = run_pipeline(dirty_ohlcv)
        assert dirty_result.quality.quality_score <= clean_result.quality.quality_score

    def test_pre_validation_fails_on_dirty_data(self, dirty_ohlcv):
        result = run_pipeline(dirty_ohlcv)
        assert result.pre_validation.valid is False

    def test_issue_manifest_non_empty_on_dirty_data(self, dirty_ohlcv):
        result = run_pipeline(dirty_ohlcv)
        assert result.quality.n_issues > 0

    def test_summary_returns_string(self, clean_ohlcv):
        result = run_pipeline(clean_ohlcv)
        summary = result.summary()
        assert isinstance(summary, str)
        assert "Grade" in summary