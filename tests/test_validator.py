# tests/test_validator.py
"""
Tests for market_pipeline.validator

Every test uses synthetic data from conftest.py — no network calls.
"""

import numpy as np
import pandas as pd
import pytest

from market_pipeline.validator import ValidationResult, assert_pipeline_invariants, validate_ohlcv


class TestValidateOhlcv:

    def test_clean_data_passes(self, clean_ohlcv):
        result = validate_ohlcv(clean_ohlcv)
        assert result.valid is True
        assert result.n_violations == 0

    def test_returns_validation_result(self, clean_ohlcv):
        result = validate_ohlcv(clean_ohlcv)
        assert isinstance(result, ValidationResult)

    def test_detects_high_lt_low(self, dirty_ohlcv):
        result = validate_ohlcv(dirty_ohlcv)
        assert result.violations["high_lt_low"] >= 1

    def test_detects_negative_volume(self, dirty_ohlcv):
        result = validate_ohlcv(dirty_ohlcv)
        assert result.violations["negative_volume"] == 1

    def test_detects_null_close(self, dirty_ohlcv):
        result = validate_ohlcv(dirty_ohlcv)
        assert result.violations["null_close"] == 1

    def test_detects_zero_close(self, dirty_ohlcv):
        result = validate_ohlcv(dirty_ohlcv)
        assert result.violations["zero_close"] == 1

    def test_dirty_data_fails(self, dirty_ohlcv):
        result = validate_ohlcv(dirty_ohlcv)
        assert result.valid is False
        assert result.n_violations > 0

    def test_detects_duplicate_dates(self, clean_ohlcv):
        duped = pd.concat([clean_ohlcv, clean_ohlcv.iloc[[0]]])
        result = validate_ohlcv(duped)
        assert result.violations["duplicate_dates"] >= 1

    def test_violation_dates_populated(self, dirty_ohlcv):
        result = validate_ohlcv(dirty_ohlcv)
        # negative_volume is at row 5 — should have exactly one date
        assert len(result.violation_dates["negative_volume"]) == 1

    def test_raises_on_missing_columns(self, clean_ohlcv):
        with pytest.raises(ValueError, match="missing required columns"):
            validate_ohlcv(clean_ohlcv.drop(columns=["High"]))

    def test_n_rows_correct(self, clean_ohlcv):
        result = validate_ohlcv(clean_ohlcv)
        assert result.n_rows == len(clean_ohlcv)

    def test_summary_contains_pass(self, clean_ohlcv):
        result = validate_ohlcv(clean_ohlcv)
        assert "PASS" in result.summary()

    def test_summary_contains_fail(self, dirty_ohlcv):
        result = validate_ohlcv(dirty_ohlcv)
        assert "FAIL" in result.summary()


class TestAssertPipelineInvariants:

    def test_clean_result_passes(self, good_pipeline_result):
        failures = assert_pipeline_invariants(good_pipeline_result)
        assert failures == []

    def test_detects_negative_close(self, clean_ohlcv):
        df = clean_ohlcv.copy()
        df.iloc[5, df.columns.get_loc("Close")] = -10.0
        result = {"clean_data": df, "quality_score": 0.95}
        failures = assert_pipeline_invariants(result)
        assert any("Non-positive" in f for f in failures)

    def test_detects_high_lt_low(self, clean_ohlcv):
        df = clean_ohlcv.copy()
        df.iloc[5, df.columns.get_loc("High")] = df.iloc[5]["Low"] - 1.0
        result = {"clean_data": df, "quality_score": 0.95}
        failures = assert_pipeline_invariants(result)
        assert any("High < Low" in f for f in failures)

    def test_detects_nan_close(self, clean_ohlcv):
        df = clean_ohlcv.copy()
        df.iloc[5, df.columns.get_loc("Close")] = float("nan")
        result = {"clean_data": df, "quality_score": 0.95}
        failures = assert_pipeline_invariants(result)
        assert any("NaN" in f for f in failures)

    def test_detects_bad_quality_score(self, good_pipeline_result):
        good_pipeline_result["quality_score"] = 1.5
        failures = assert_pipeline_invariants(good_pipeline_result)
        assert any("quality_score" in f for f in failures)

    def test_detects_high_outlier_rate(self, clean_ohlcv):
        df = clean_ohlcv.copy()
        df["is_outlier"] = True          # 100% outlier rate
        result = {"clean_data": df, "quality_score": 0.95}
        failures = assert_pipeline_invariants(result)
        assert any("Outlier rate" in f for f in failures)

    def test_empty_dataframe_fails(self):
        result = {"clean_data": pd.DataFrame(), "quality_score": 0.95}
        failures = assert_pipeline_invariants(result)
        assert len(failures) > 0