# tests/test_scoring.py
"""Tests for market_pipeline.scoring"""

import pandas as pd
import pytest

from market_pipeline.scoring import (
    IssueRecord,
    QualityScore,
    ScoreComponents,
    assign_grade,
    build_issue_manifest,
    compute_completeness_score,
    compute_outlier_score,
    compute_quality_score,
    compute_validity_score,
)
from market_pipeline.validator import validate_ohlcv


class TestAssignGrade:

    def test_perfect_score_is_A(self):
        assert assign_grade(1.0) == "A"

    def test_boundary_A(self):
        assert assign_grade(0.99) == "A"

    def test_boundary_B(self):
        assert assign_grade(0.95) == "B"

    def test_boundary_C(self):
        assert assign_grade(0.90) == "C"

    def test_boundary_D(self):
        assert assign_grade(0.80) == "D"

    def test_below_D_is_F(self):
        assert assign_grade(0.79) == "F"

    def test_zero_is_F(self):
        assert assign_grade(0.0) == "F"

    def test_all_grades_are_valid_letters(self):
        scores = [1.0, 0.97, 0.92, 0.85, 0.50, 0.0]
        for score in scores:
            assert assign_grade(score) in {"A", "B", "C", "D", "F"}


class TestComputeValidityScore:

    def test_clean_data_scores_one(self, clean_ohlcv):
        validation = validate_ohlcv(clean_ohlcv)
        score = compute_validity_score(validation)
        assert score == 1.0

    def test_score_in_unit_interval(self, dirty_ohlcv):
        validation = validate_ohlcv(dirty_ohlcv)
        score = compute_validity_score(validation)
        assert 0.0 <= score <= 1.0

    def test_dirty_data_scores_less_than_one(self, dirty_ohlcv):
        validation = validate_ohlcv(dirty_ohlcv)
        score = compute_validity_score(validation)
        assert score < 1.0

    def test_score_never_negative(self, dirty_ohlcv):
        """Even if many checks fail on same rows, score >= 0."""
        validation = validate_ohlcv(dirty_ohlcv)
        score = compute_validity_score(validation)
        assert score >= 0.0


class TestComputeCompletenessScore:

    def test_no_missing_scores_one(self):
        assert compute_completeness_score(n_rows=100, n_missing=0) == 1.0

    def test_all_missing_scores_zero(self):
        assert compute_completeness_score(n_rows=100, n_missing=100) == 0.0

    def test_half_missing_scores_half(self):
        score = compute_completeness_score(n_rows=100, n_missing=50)
        assert abs(score - 0.5) < 1e-9

    def test_zero_rows_scores_one(self):
        assert compute_completeness_score(n_rows=0, n_missing=0) == 1.0


class TestComputeOutlierScore:

    def test_no_outliers_scores_one(self):
        assert compute_outlier_score(n_returns=500, n_outliers=0) == 1.0

    def test_all_outliers_scores_zero(self):
        assert compute_outlier_score(n_returns=100, n_outliers=100) == 0.0

    def test_zero_returns_scores_one(self):
        assert compute_outlier_score(n_returns=0, n_outliers=0) == 1.0

    def test_score_in_unit_interval(self):
        score = compute_outlier_score(n_returns=500, n_outliers=12)
        assert 0.0 <= score <= 1.0


class TestScoreComponents:

    def test_raises_on_value_above_one(self):
        with pytest.raises(ValueError, match="validity_score"):
            ScoreComponents(
                validity_score=1.5,
                completeness_score=1.0,
                outlier_score=1.0,
            )

    def test_raises_on_negative_value(self):
        with pytest.raises(ValueError, match="outlier_score"):
            ScoreComponents(
                validity_score=1.0,
                completeness_score=1.0,
                outlier_score=-0.1,
            )

    def test_valid_construction(self):
        c = ScoreComponents(
            validity_score=0.98,
            completeness_score=0.99,
            outlier_score=1.0,
        )
        assert c.validity_score == 0.98


class TestBuildIssueManifest:

    def test_structural_violations_appear(self, dirty_ohlcv):
        validation = validate_ohlcv(dirty_ohlcv)
        issues = build_issue_manifest(validation)
        types = [i.issue_type for i in issues]
        assert "structural_violation" in types

    def test_issues_sorted_chronologically(self, dirty_ohlcv):
        validation = validate_ohlcv(dirty_ohlcv)
        issues = build_issue_manifest(validation)
        dates = [i.date for i in issues]
        assert dates == sorted(dates)

    def test_clean_data_has_no_structural_issues(self, clean_ohlcv):
        validation = validate_ohlcv(clean_ohlcv)
        issues = build_issue_manifest(validation)
        structural = [i for i in issues if i.issue_type == "structural_violation"]
        assert structural == []

    def test_issue_record_str(self, dirty_ohlcv):
        validation = validate_ohlcv(dirty_ohlcv)
        issues = build_issue_manifest(validation)
        assert len(str(issues[0])) > 0


class TestComputeQualityScore:

    def test_returns_quality_score_object(self, clean_ohlcv):
        validation = validate_ohlcv(clean_ohlcv)
        qs = compute_quality_score(
            validation=validation,
            n_rows=len(clean_ohlcv),
            n_missing=0,
            n_returns=len(clean_ohlcv) - 1,
            n_outliers=0,
        )
        assert isinstance(qs, QualityScore)

    def test_perfect_data_gets_grade_A(self, clean_ohlcv):
        validation = validate_ohlcv(clean_ohlcv)
        qs = compute_quality_score(
            validation=validation,
            n_rows=len(clean_ohlcv),
            n_missing=0,
            n_returns=len(clean_ohlcv) - 1,
            n_outliers=0,
        )
        assert qs.grade == "A"
        assert qs.quality_score >= 0.99

    def test_score_is_product_of_components(self, clean_ohlcv):
        validation = validate_ohlcv(clean_ohlcv)
        qs = compute_quality_score(
            validation=validation,
            n_rows=100,
            n_missing=5,
            n_returns=95,
            n_outliers=2,
        )
        expected = (
            qs.components.validity_score
            * qs.components.completeness_score
            * qs.components.outlier_score
        )
        assert abs(qs.quality_score - round(expected, 4)) < 1e-4