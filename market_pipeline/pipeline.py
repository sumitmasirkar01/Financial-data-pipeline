# market_pipeline/pipeline.py
"""
Main orchestrator for the market data quality pipeline.

This module contains no logic of its own. It delegates every step to
the appropriate module and assembles the results into a PipelineResult.

Pipeline steps
--------------
1. validate        — structural checks on raw input  (validator)
2. adjust          — split and dividend back-adjustment  (adjustments)
3. fill_gaps       — reindex to trading calendar, forward-fill  (gaps)
4. compute_returns — log returns, simple returns, rolling vol  (returns)
5. detect_outliers — rolling MAD modified z-score  (outliers)
6. score           — quality score, grade, issue manifest  (scoring)
7. check           — post-pipeline invariants  (validator)

Usage
-----
Single ticker (you already have the DataFrame):

    from market_pipeline.pipeline import run_pipeline
    result = run_pipeline(df)
    print(result.quality.summary())

Multiple tickers (fetched automatically):

    from market_pipeline.pipeline import run_pipeline_for_tickers
    results = run_pipeline_for_tickers(['AAPL', 'MSFT', 'RELIANCE.NS'])
    for ticker, result in results.items():
        print(ticker, result.quality.grade)
"""

import logging
from dataclasses import dataclass, field
from datetime import datetime, timezone
from typing import Dict, List, Optional

import pandas as pd

from market_pipeline import __version__
from market_pipeline.adjustments import (
    AdjustmentReport,
    DividendEvent,
    SplitEvent,
    apply_corporate_actions,
)
from market_pipeline.config import (
    DEFAULT_PERIOD,
    OUTLIER_MIN_PERIODS,
    OUTLIER_THRESHOLD,
    OUTLIER_WINDOW,
)
from market_pipeline.fetcher import fetch_multiple, fetch_ohlcv
from market_pipeline.gaps import FillReport, fill_gaps
from market_pipeline.outliers import OutlierResult, detect_outliers_mad
from market_pipeline.returns import ReturnStats, compute_return_stats, compute_returns
from market_pipeline.scoring import QualityScore, compute_quality_score
from market_pipeline.validator import ValidationResult, assert_pipeline_invariants, validate_ohlcv

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Result type
# ---------------------------------------------------------------------------

@dataclass
class PipelineResult:
    """
    Complete output from a single run_pipeline() call.

    Attributes
    ----------
    clean_data         : pd.DataFrame
        Fully processed OHLCV DataFrame with return and outlier columns added.
    quality            : QualityScore
        Combined score, grade, component scores, and issue manifest.
    pre_validation     : ValidationResult
        Structural check on the raw input before any cleaning.
    adjustment_report  : AdjustmentReport
        Record of every split and dividend applied (or skipped).
    fill_report        : FillReport
        Record of every cell forward-filled during gap correction.
    outlier_result     : OutlierResult or None
        Full rolling MAD result. None if data was too short for detection.
    return_stats       : ReturnStats or None
        Annualised summary statistics. None if returns could not be computed.
    invariant_failures : list of str
        Post-pipeline invariant violations. Should always be empty.
        Non-empty means there is a bug in the pipeline itself.
    metadata           : dict
        Pipeline version, timestamp, and key parameters used.
    """
    clean_data:         pd.DataFrame
    quality:            QualityScore
    pre_validation:     ValidationResult
    adjustment_report:  AdjustmentReport
    fill_report:        FillReport
    outlier_result:     Optional[OutlierResult]  = None
    return_stats:       Optional[ReturnStats]     = None
    invariant_failures: List[str]                 = field(default_factory=list)
    metadata:           Dict                      = field(default_factory=dict)

    def summary(self) -> str:
        lines = [
            "=" * 60,
            f"PIPELINE RESULT",
            "=" * 60,
            self.quality.summary(),
            "-" * 60,
            f"Pre-validation  : {'PASS' if self.pre_validation.valid else 'FAIL'}  "
            f"({self.pre_validation.n_violations} violation(s))",
            f"Splits applied  : {self.adjustment_report.n_splits}",
            f"Dividends applied: {self.adjustment_report.n_dividends}",
            f"Rows gap-filled : {self.fill_report.n_rows_filled}",
            f"Outliers flagged: {self.outlier_result.n_outliers if self.outlier_result else 'N/A'}",
            f"Total rows      : {len(self.clean_data)}",
        ]
        if self.invariant_failures:
            lines.append("INVARIANT FAILURES (pipeline bug):")
            for f in self.invariant_failures:
                lines.append(f"  !! {f}")
        if self.return_stats:
            lines.append("-" * 60)
            lines.append(self.return_stats.summary())
        lines.append("=" * 60)
        return "\n".join(lines)


# ---------------------------------------------------------------------------
# Core pipeline
# ---------------------------------------------------------------------------

def run_pipeline(
    df: pd.DataFrame,
    splits:             Optional[List[SplitEvent]]    = None,
    dividends:          Optional[List[DividendEvent]] = None,
    outlier_window:     int   = OUTLIER_WINDOW,
    outlier_threshold:  float = OUTLIER_THRESHOLD,
    outlier_min_periods:int   = OUTLIER_MIN_PERIODS,
    calendar:           Optional[str] = None,
) -> PipelineResult:
    """
    Run the complete data quality pipeline on a single OHLCV DataFrame.

    Parameters
    ----------
    df : pd.DataFrame
        Raw OHLCV DataFrame with DatetimeIndex.
        Fetch it with fetcher.fetch_ohlcv() or provide your own.
    splits : list of SplitEvent, optional
        Stock split events to back-adjust for.
    dividends : list of DividendEvent, optional
        Cash dividend events to back-adjust for.
    outlier_window : int
        Rolling window size for MAD outlier detection (trading days).
    outlier_threshold : float
        Modified z-score threshold for outlier flagging.
    outlier_min_periods : int
        Minimum observations in a rolling window before scoring.
    calendar : str or None
        pandas_market_calendars exchange name (e.g. 'XNYS', 'XNSE').
        None → generic Mon–Fri business days.

    Returns
    -------
    PipelineResult
        Complete structured output. See class docstring for all fields.

    Raises
    ------
    ValueError
        If df is empty or missing required OHLCV columns.
    """
    if df is None or df.empty:
        raise ValueError("Input DataFrame is empty.")

    logger.info(
        "run_pipeline: starting  rows=%d  splits=%d  dividends=%d",
        len(df),
        len(splits) if splits else 0,
        len(dividends) if dividends else 0,
    )

    working = df.copy()

    # ── Step 1: Pre-pipeline structural validation ─────────────────────────
    logger.info("Step 1/7 — structural validation")
    pre_validation = validate_ohlcv(working)
    if not pre_validation.valid:
        logger.warning(
            "Pre-validation: %d violation(s) found — pipeline will continue "
            "but data quality may be affected.",
            pre_validation.n_violations,
        )

    # ── Step 2: Corporate action adjustments ──────────────────────────────
    logger.info("Step 2/7 — corporate action adjustments")
    working, adjustment_report = apply_corporate_actions(
        working, splits=splits, dividends=dividends
    )

    # ── Step 3: Gap filling ───────────────────────────────────────────────
    logger.info("Step 3/7 — gap detection and filling")
    working, detected_gaps, fill_report = fill_gaps(working, calendar=calendar)
    n_missing = fill_report.n_rows_filled

    # ── Step 4: Returns ───────────────────────────────────────────────────
    logger.info("Step 4/7 — computing returns")
    working = compute_returns(working)

    # ── Step 5: Outlier detection ─────────────────────────────────────────
    logger.info("Step 5/7 — outlier detection")
    log_returns = working["log_return"].dropna()
    outlier_result: Optional[OutlierResult] = None
    n_outliers = 0

    if len(log_returns) > outlier_window:
        outlier_result = detect_outliers_mad(
            log_returns,
            window=outlier_window,
            threshold=outlier_threshold,
            min_periods=outlier_min_periods,
        )
        n_outliers = outlier_result.n_outliers

        # Stamp outlier flags onto the working DataFrame
        working["is_outlier"] = False
        working.loc[outlier_result.is_outlier.index, "is_outlier"] = (
            outlier_result.is_outlier
        )
        working["modified_z"] = float("nan")
        working.loc[outlier_result.modified_z.index, "modified_z"] = (
            outlier_result.modified_z
        )
    else:
        logger.warning(
            "Too few returns (%d) for outlier detection (window=%d). Skipping.",
            len(log_returns), outlier_window,
        )

    # ── Step 6: Quality scoring ───────────────────────────────────────────
    logger.info("Step 6/7 — quality scoring")
    quality = compute_quality_score(
        validation=pre_validation,
        n_rows=len(working),
        n_missing=n_missing,
        n_returns=len(log_returns),
        n_outliers=n_outliers,
        fill_report=fill_report,
        outlier_result=outlier_result,
    )

    # ── Step 7: Post-pipeline invariant check ─────────────────────────────
    logger.info("Step 7/7 — post-pipeline invariant check")
    invariant_failures = assert_pipeline_invariants({
        "clean_data":    working,
        "quality_score": quality.quality_score,
    })
    if invariant_failures:
        logger.error(
            "Post-pipeline invariants FAILED — this is a pipeline bug:\n%s",
            "\n".join(invariant_failures),
        )

    # ── Return statistics (best-effort) ───────────────────────────────────
    return_stats: Optional[ReturnStats] = None
    try:
        return_stats = compute_return_stats(
            log_returns=working["log_return"],
            simple_returns=working["simple_return"],
        )
    except ValueError as exc:
        logger.warning("Could not compute return stats: %s", exc)

    # ── Metadata ──────────────────────────────────────────────────────────
    metadata = {
        "pipeline_version":    __version__,
        "timestamp":           datetime.now(timezone.utc).isoformat().replace("+00:00", "Z"),
        "outlier_window":      outlier_window,
        "outlier_threshold":   outlier_threshold,
        "calendar":            calendar or "bdate_range (generic)",
        "n_splits":            adjustment_report.n_splits,
        "n_dividends":         adjustment_report.n_dividends,
    }

    result = PipelineResult(
        clean_data=working,
        quality=quality,
        pre_validation=pre_validation,
        adjustment_report=adjustment_report,
        fill_report=fill_report,
        outlier_result=outlier_result,
        return_stats=return_stats,
        invariant_failures=invariant_failures,
        metadata=metadata,
    )

    logger.info("run_pipeline complete.\n%s", result.summary())
    return result


# ---------------------------------------------------------------------------
# Multi-ticker convenience wrapper
# ---------------------------------------------------------------------------

def run_pipeline_for_tickers(
    tickers:            List[str],
    period:             str  = DEFAULT_PERIOD,
    splits_map:         Optional[Dict[str, List[SplitEvent]]]    = None,
    dividends_map:      Optional[Dict[str, List[DividendEvent]]] = None,
    outlier_window:     int   = OUTLIER_WINDOW,
    outlier_threshold:  float = OUTLIER_THRESHOLD,
    calendar:           Optional[str] = None,
) -> Dict[str, PipelineResult]:
    """
    Fetch and run the pipeline for a list of tickers.

    Failed fetches are skipped (logged as errors). Failed pipeline runs
    on individual tickers are also isolated — one bad ticker does not
    abort the rest.

    Parameters
    ----------
    tickers : list of str
        Ticker symbols to process.
    period : str
        yfinance look-back period (e.g. '2y').
    splits_map : dict, optional
        Per-ticker split events. {ticker: [SplitEvent, ...]}.
    dividends_map : dict, optional
        Per-ticker dividend events. {ticker: [DividendEvent, ...]}.
    outlier_window : int
        Rolling window for MAD detection.
    outlier_threshold : float
        Modified z-score threshold.
    calendar : str or None
        Exchange calendar name for gap detection.

    Returns
    -------
    dict
        {ticker: PipelineResult} for every ticker that succeeded.
    """
    splits_map    = splits_map    or {}
    dividends_map = dividends_map or {}

    # Fetch all tickers — failures are isolated inside fetch_multiple
    raw_data = fetch_multiple(tickers, period=period)

    results: Dict[str, PipelineResult] = {}
    failed:  List[str] = []

    for ticker, df in raw_data.items():
        try:
            results[ticker] = run_pipeline(
                df,
                splits=splits_map.get(ticker),
                dividends=dividends_map.get(ticker),
                outlier_window=outlier_window,
                outlier_threshold=outlier_threshold,
                calendar=calendar,
            )
        except Exception as exc:
            logger.error("Pipeline failed for %s: %s", ticker, exc)
            failed.append(ticker)

    if failed:
        logger.warning(
            "%d ticker(s) failed the pipeline and were excluded: %s",
            len(failed), failed,
        )

    logger.info(
        "run_pipeline_for_tickers: %d/%d succeeded",
        len(results), len(tickers),
    )
    return results