# market_pipeline/scoring.py
"""
Quality scoring and grading for pipeline output.

How the score is computed
--------------------------
Three independent component scores are multiplied together:

    quality_score = validity × completeness × outlier_score

Each component is in [0, 1]. Multiplication is intentional — a dataset
that is 100% structurally valid but 20% missing data should not score
near 1.0. The penalty compounds across dimensions.

Component definitions
---------------------
validity_score
    Fraction of *rows* that are free of structural violations.
    Note: a single row can fail multiple checks (e.g. Low > High AND
    negative volume). We count violating rows, not total check failures,
    so the score cannot go below 0.

completeness_score
    Fraction of rows that had data on the original trading calendar
    before any gap-filling. Rows added by reindex_to_calendar() are
    gaps; they reduce this score.

outlier_score
    Fraction of return observations not flagged as outliers by the
    rolling MAD detector. A handful of outliers in 500 returns still
    earns a high score — outliers are expected in real markets.

Grading
-------
Grade boundaries live in config.GRADE_BOUNDARIES. Changing the boundaries
there automatically changes the grading here — no edits to this file needed.

Issue manifest
--------------
build_issue_manifest() assembles a structured list of every anomaly found,
with dates and types. This is the machine-readable audit trail that a raw
letter grade doesn't provide.
"""

import logging
from dataclasses import dataclass, field
from typing import Dict, List, Optional

import pandas as pd

from market_pipeline.config import GRADE_BOUNDARIES
from market_pipeline.gaps import FillReport
from market_pipeline.outliers import OutlierResult
from market_pipeline.validator import ValidationResult

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Return types
# ---------------------------------------------------------------------------

@dataclass
class ScoreComponents:
    """
    The three independent scores that feed into quality_score.

    All values are floats in [0, 1].
    """
    validity_score:     float   # rows without violations / total rows
    completeness_score: float   # rows with original data / total rows
    outlier_score:      float   # non-outlier returns / total returns

    def __post_init__(self) -> None:
        for name in ("validity_score", "completeness_score", "outlier_score"):
            val = getattr(self, name)
            if not (0.0 <= val <= 1.0):
                raise ValueError(
                    f"ScoreComponents.{name} must be in [0, 1], got {val:.4f}."
                )


@dataclass
class IssueRecord:
    """
    A single anomaly found during pipeline processing.

    Attributes
    ----------
    date      : the date the issue occurred
    issue_type: category string (e.g. 'structural_violation', 'gap', 'outlier')
    detail    : human-readable description of the specific issue
    """
    date: pd.Timestamp
    issue_type: str
    detail: str

    def __str__(self) -> str:
        return f"{self.date.strftime('%Y-%m-%d')}  [{self.issue_type}]  {self.detail}"


@dataclass
class QualityScore:
    """
    Full quality assessment for a single ticker's pipeline run.

    Attributes
    ----------
    quality_score : float
        Combined score in [0, 1].  quality_score = validity × completeness × outlier.
    grade         : str
        Letter grade derived from GRADE_BOUNDARIES in config. One of A B C D F.
    components    : ScoreComponents
        The three individual component scores.
    issues        : list of IssueRecord
        Machine-readable list of every anomaly found with timestamps.
    """
    quality_score: float
    grade:         str
    components:    ScoreComponents
    issues:        List[IssueRecord] = field(default_factory=list)

    @property
    def n_issues(self) -> int:
        return len(self.issues)

    def summary(self) -> str:
        lines = [
            f"Grade          : {self.grade}  ({self.quality_score:.4f})",
            f"Validity       : {self.components.validity_score:.4f}",
            f"Completeness   : {self.components.completeness_score:.4f}",
            f"Outlier score  : {self.components.outlier_score:.4f}",
            f"Total issues   : {self.n_issues}",
        ]
        if self.issues:
            lines.append("First 10 issues:")
            for issue in self.issues[:10]:
                lines.append(f"  {issue}")
            if self.n_issues > 10:
                lines.append(f"  ... and {self.n_issues - 10} more")
        return "\n".join(lines)


# ---------------------------------------------------------------------------
# Component score calculators
# ---------------------------------------------------------------------------

def compute_validity_score(validation: ValidationResult) -> float:
    """
    Fraction of rows with no structural violations.

    Uses unique violating rows, not total check failures.
    A row failing 3 checks is counted once, not three times.
    This prevents the score from going below zero.

    Parameters
    ----------
    validation : ValidationResult
        Output of validator.validate_ohlcv().

    Returns
    -------
    float in [0, 1]
    """
    if validation.n_rows == 0:
        return 1.0

    # Collect all dates that appear in any violation, then count unique ones
    all_violating_dates: set = set()
    for dates in validation.violation_dates.values():
        all_violating_dates.update(dates)

    n_clean_rows = validation.n_rows - len(all_violating_dates)
    score = max(0.0, n_clean_rows / validation.n_rows)

    logger.debug(
        "validity_score: %d/%d clean rows → %.4f",
        n_clean_rows, validation.n_rows, score,
    )
    return score


def compute_completeness_score(n_rows: int, n_missing: int) -> float:
    """
    Fraction of rows that had real data (not gap-filled).

    Parameters
    ----------
    n_rows    : total rows in the reindexed DataFrame
    n_missing : rows that were NaN before filling (from FillReport)

    Returns
    -------
    float in [0, 1]
    """
    if n_rows == 0:
        return 1.0
    score = max(0.0, (n_rows - n_missing) / n_rows)
    logger.debug(
        "completeness_score: %d/%d real rows → %.4f",
        n_rows - n_missing, n_rows, score,
    )
    return score


def compute_outlier_score(n_returns: int, n_outliers: int) -> float:
    """
    Fraction of return observations not flagged as outliers.

    Parameters
    ----------
    n_returns  : total non-NaN return observations assessed
    n_outliers : observations flagged by detect_outliers_mad

    Returns
    -------
    float in [0, 1]
    """
    if n_returns == 0:
        return 1.0
    score = max(0.0, (n_returns - n_outliers) / n_returns)
    logger.debug(
        "outlier_score: %d/%d clean returns → %.4f",
        n_returns - n_outliers, n_returns, score,
    )
    return score


# ---------------------------------------------------------------------------
# Grading
# ---------------------------------------------------------------------------

def assign_grade(quality_score: float) -> str:
    """
    Convert a quality_score float to a letter grade.

    Boundaries are read from config.GRADE_BOUNDARIES. Change them
    there — this function needs no edits.

    Parameters
    ----------
    quality_score : float in [0, 1]

    Returns
    -------
    str — one of 'A', 'B', 'C', 'D', 'F'
    """
    # GRADE_BOUNDARIES is ordered A → D (highest → lowest threshold)
    for grade, boundary in sorted(
        GRADE_BOUNDARIES.items(), key=lambda x: x[1], reverse=True
    ):
        if quality_score >= boundary:
            return grade
    return "F"


# ---------------------------------------------------------------------------
# Issue manifest builder
# ---------------------------------------------------------------------------

def build_issue_manifest(
    validation:     ValidationResult,
    fill_report:    Optional[FillReport]  = None,
    outlier_result: Optional[OutlierResult] = None,
) -> List[IssueRecord]:
    """
    Assemble a time-stamped list of every anomaly found during the pipeline.

    This is the machine-readable audit trail. Store it in a database,
    export it to CSV, or pipe it to an alerting system.

    Parameters
    ----------
    validation     : from validator.validate_ohlcv() on the raw input
    fill_report    : from gaps.forward_fill_prices() — gap-filled rows
    outlier_result : from outliers.detect_outliers_mad() — flagged returns

    Returns
    -------
    list of IssueRecord, sorted chronologically
    """
    issues: List[IssueRecord] = []

    # ── Structural violations ─────────────────────────────────────────────
    for check_name, dates in validation.violation_dates.items():
        for date in dates:
            issues.append(IssueRecord(
                date=pd.Timestamp(date),
                issue_type="structural_violation",
                detail=check_name,
            ))

    # ── Gap-filled rows ───────────────────────────────────────────────────
    if fill_report is not None:
        for date in fill_report.filled_dates:
            issues.append(IssueRecord(
                date=pd.Timestamp(date),
                issue_type="gap_filled",
                detail="row was missing and forward-filled",
            ))

    # ── Outliers ──────────────────────────────────────────────────────────
    if outlier_result is not None:
        for date in outlier_result.outlier_dates:
            mz = outlier_result.modified_z.get(date, float("nan"))
            issues.append(IssueRecord(
                date=pd.Timestamp(date),
                issue_type="outlier",
                detail=f"modified z-score = {mz:+.2f}",
            ))

    issues.sort(key=lambda r: r.date)

    logger.info("build_issue_manifest: %d total issues recorded", len(issues))
    return issues


# ---------------------------------------------------------------------------
# Orchestrator
# ---------------------------------------------------------------------------

def compute_quality_score(
    validation:      ValidationResult,
    n_rows:          int,
    n_missing:       int,
    n_returns:       int,
    n_outliers:      int,
    fill_report:     Optional[FillReport]   = None,
    outlier_result:  Optional[OutlierResult] = None,
) -> QualityScore:
    """
    Compute the combined quality score, grade, and issue manifest.

    This is the function pipeline.py calls. The individual component
    calculators above are available for isolated testing.

    Parameters
    ----------
    validation     : ValidationResult from pre-pipeline validate_ohlcv()
    n_rows         : total rows in the working DataFrame
    n_missing      : rows filled by the gap module
    n_returns      : total non-NaN return observations
    n_outliers     : outliers flagged by the MAD detector
    fill_report    : FillReport for the issue manifest (optional)
    outlier_result : OutlierResult for the issue manifest (optional)

    Returns
    -------
    QualityScore
    """
    components = ScoreComponents(
        validity_score=     compute_validity_score(validation),
        completeness_score= compute_completeness_score(n_rows, n_missing),
        outlier_score=      compute_outlier_score(n_returns, n_outliers),
    )

    quality_score = round(
        components.validity_score
        * components.completeness_score
        * components.outlier_score,
        4,
    )

    grade  = assign_grade(quality_score)
    issues = build_issue_manifest(validation, fill_report, outlier_result)

    result = QualityScore(
        quality_score=quality_score,
        grade=grade,
        components=components,
        issues=issues,
    )

    logger.info("compute_quality_score:\n%s", result.summary())
    return result