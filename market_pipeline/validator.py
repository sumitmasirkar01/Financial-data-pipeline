# market_pipeline/validator.py
"""
Structural validation for OHLCV DataFrames.

Two responsibilities:
  1. Pre-pipeline  — validate_ohlcv() checks raw data before any cleaning.
  2. Post-pipeline — assert_pipeline_invariants() re-checks after cleaning
                     to catch bugs introduced by the pipeline itself.

Nothing here modifies data. These functions are read-only inspectors.
"""

import logging
from dataclasses import dataclass, field
from typing import Dict, List

import numpy as np
import pandas as pd

from market_pipeline.config import MAX_EXPECTED_OUTLIER_RATE

logger = logging.getLogger(__name__)

# Columns that must be present for any validation to make sense
_REQUIRED_COLUMNS = ["Open", "High", "Low", "Close", "Volume"]


# ---------------------------------------------------------------------------
# Return type
# ---------------------------------------------------------------------------

@dataclass
class ValidationResult:
    """
    Structured result from validate_ohlcv().

    Attributes
    ----------
    valid : bool
        True only when there are zero violations across all checks.
    n_rows : int
        Total number of rows inspected.
    n_violations : int
        Sum of all individual violation counts.
    violations : dict
        Per-check violation counts. Zero means that check passed.
    violation_dates : dict
        Per-check list of dates where a violation was found.
        Useful for debugging and the issue manifest.
    """
    valid: bool
    n_rows: int
    n_violations: int
    violations: Dict[str, int] = field(default_factory=dict)
    violation_dates: Dict[str, List] = field(default_factory=dict)

    def summary(self) -> str:
        """One-line human-readable summary."""
        if self.valid:
            return f"PASS  ({self.n_rows} rows, 0 violations)"
        lines = [f"FAIL  ({self.n_rows} rows, {self.n_violations} violations)"]
        for check, count in self.violations.items():
            if count > 0:
                lines.append(f"  {check}: {count}")
        return "\n".join(lines)


# ---------------------------------------------------------------------------
# Internal helpers
# ---------------------------------------------------------------------------

def _check_missing_columns(df: pd.DataFrame) -> None:
    """Raise immediately if required columns are absent."""
    missing = set(_REQUIRED_COLUMNS) - set(df.columns)
    if missing:
        raise ValueError(
            f"DataFrame is missing required columns: {sorted(missing)}. "
            f"Got: {sorted(df.columns.tolist())}"
        )


def _dates_where(mask: pd.Series) -> List:
    """Return index values where mask is True, as a plain list."""
    return mask.index[mask].tolist()


# ---------------------------------------------------------------------------
# Pre-pipeline validator
# ---------------------------------------------------------------------------

def validate_ohlcv(df: pd.DataFrame) -> ValidationResult:
    """
    Check structural constraints on a raw OHLCV DataFrame.

    Checks performed
    ----------------
    Price logic
      low_gt_open      — Low > Open      (impossible in a valid bar)
      low_gt_close     — Low > Close     (impossible in a valid bar)
      open_gt_high     — Open > High     (impossible in a valid bar)
      close_gt_high    — Close > High    (impossible in a valid bar)
      high_lt_low      — High < Low      (direct inversion — worst violation)
    Price values
      zero_close       — Close == 0      (almost always a data error)
      negative_close   — Close < 0       (price can never be negative)
      null_open        — Open is NaN
      null_high        — High is NaN
      null_low         — Low is NaN
      null_close       — Close is NaN
      inf_close        — Close is Inf    (corrupted numeric float)
    Volume
      negative_volume  — Volume < 0      (impossible physically)
      null_volume      — Volume is NaN
      inf_volume       — Volume is Inf   (corrupted numeric float)
    Index
      duplicate_dates  — Repeated dates in the index

    Parameters
    ----------
    df : pd.DataFrame
        Raw OHLCV DataFrame with a DatetimeIndex.

    Returns
    -------
    ValidationResult
        Structured result with per-check counts and dates.

    Raises
    ------
    ValueError
        If required columns are missing entirely.
    """
    _check_missing_columns(df)

    violations: Dict[str, int] = {}
    violation_dates: Dict[str, List] = {}

    def _record(name: str, mask: pd.Series) -> None:
        count = int(mask.sum())
        violations[name] = count
        violation_dates[name] = _dates_where(mask) if count > 0 else []

    # ── Price logic ───────────────────────────────────────────────────────
    _record("high_lt_low",    df["High"] < df["Low"])
    _record("low_gt_open",    df["Low"]  > df["Open"])
    _record("low_gt_close",   df["Low"]  > df["Close"])
    _record("open_gt_high",   df["Open"] > df["High"])
    _record("close_gt_high",  df["Close"] > df["High"])

    # ── Price values ──────────────────────────────────────────────────────
    _record("zero_close",     df["Close"] == 0)
    _record("negative_close", df["Close"] < 0)
    _record("null_open",      df["Open"].isna())
    _record("null_high",      df["High"].isna())
    _record("null_low",       df["Low"].isna())
    _record("null_close",     df["Close"].isna())
    _record("inf_close",      np.isinf(df["Close"]))

    # ── Volume ────────────────────────────────────────────────────────────
    _record("negative_volume", df["Volume"] < 0)
    _record("null_volume",     df["Volume"].isna())
    _record("inf_volume",      np.isinf(df["Volume"]))

    # ── Index integrity ───────────────────────────────────────────────────
    dup_mask = pd.Series(df.index.duplicated(), index=df.index)
    _record("duplicate_dates", dup_mask)

    n_violations = sum(violations.values())
    result = ValidationResult(
        valid=n_violations == 0,
        n_rows=len(df),
        n_violations=n_violations,
        violations=violations,
        violation_dates=violation_dates,
    )

    if result.valid:
        logger.debug("validate_ohlcv: PASS (%d rows)", len(df))
    else:
        logger.warning("validate_ohlcv: %d violation(s) found\n%s",
                       n_violations, result.summary())

    return result


# ---------------------------------------------------------------------------
# Post-pipeline invariant checker
# ---------------------------------------------------------------------------

def assert_pipeline_invariants(pipeline_result: Dict) -> List[str]:
    """
    Verify that the pipeline output satisfies all structural guarantees.

    Run this after run_quality_pipeline(). It answers the question:
    "Did the cleaning process itself introduce any new problems?"

    Parameters
    ----------
    pipeline_result : dict
        The dict returned by run_quality_pipeline().
        Must contain 'clean_data' (DataFrame) and 'quality_score' (float).

    Returns
    -------
    list of str
        Each string is a violated invariant. Empty list means all clear.
    """
    failures: List[str] = []
    df: pd.DataFrame = pipeline_result.get("clean_data")

    if df is None or df.empty:
        failures.append("clean_data is missing or empty")
        return failures

    # ── Structural ────────────────────────────────────────────────────────
    if (df["High"] < df["Low"]).any():
        n = int((df["High"] < df["Low"]).sum())
        failures.append(f"High < Low on {n} row(s) after cleaning")

    if (df["Close"] <= 0).any():
        n = int((df["Close"] <= 0).sum())
        failures.append(f"Non-positive Close price on {n} row(s) after cleaning")

    if df["Close"].isna().any():
        n = int(df["Close"].isna().sum())
        failures.append(f"{n} NaN Close value(s) remain after forward-fill")

    # ── Outlier rate sanity ───────────────────────────────────────────────
    if "is_outlier" in df.columns:
        rate = float(df["is_outlier"].mean())
        if rate > MAX_EXPECTED_OUTLIER_RATE:
            failures.append(
                f"Outlier rate {rate:.1%} exceeds expected cap of "
                f"{MAX_EXPECTED_OUTLIER_RATE:.0%} — check threshold config"
            )

    # ── Score bounds ──────────────────────────────────────────────────────
    score = pipeline_result.get("quality_score")
    if score is not None and not (0.0 <= score <= 1.0):
        failures.append(f"quality_score {score} is outside [0, 1]")

    # ── Post-pipeline structural re-validation ────────────────────────────
    post = validate_ohlcv(df[_REQUIRED_COLUMNS])
    if not post.valid:
        failures.append(
            f"Post-pipeline validate_ohlcv found {post.n_violations} "
            "violation(s) — the pipeline introduced bad data"
        )

    if failures:
        for f in failures:
            logger.error("Invariant violation: %s", f)
    else:
        logger.debug("assert_pipeline_invariants: all clear")

    return failures