# market_pipeline/outliers.py
"""
Outlier detection for financial return series.

Why MAD instead of standard deviation
--------------------------------------
Financial returns have fat tails (excess kurtosis >> 0). A single extreme
event inflates the standard deviation enough to mask dozens of real anomalies
— this is called the masking effect. The Median Absolute Deviation (MAD) is
resistant to this because medians are not pulled by extreme values.

Two modes are provided:
  - Global  : compute_mad / global_modified_z_scores
              Uses the full series. Fast. Good for exploratory comparison.
  - Rolling : detect_outliers_mad
              Uses a sliding window (~3 months). Adapts to changing volatility
              regimes (e.g. COVID crash followed by calm). This is the mode
              used by the pipeline.

The 1.4826 consistency factor
------------------------------
Under a normal distribution, MAD * 1.4826 ≈ σ (standard deviation).
This makes the modified z-scores comparable to classic z-scores, so the
threshold of 3.5 carries the same intuition as "3.5 standard deviations away".
"""

import logging
from dataclasses import dataclass, field
from typing import List

import pandas as pd
import numpy as np

from market_pipeline.config import (
    MAD_CONSISTENCY_FACTOR,
    OUTLIER_MIN_PERIODS,
    OUTLIER_THRESHOLD,
    OUTLIER_WINDOW,
)

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Return type
# ---------------------------------------------------------------------------

@dataclass
class OutlierResult:
    """
    Result from detect_outliers_mad().

    Attributes
    ----------
    modified_z : pd.Series
        Rolling modified z-score for every date.
        NaN where the window has fewer than min_periods observations,
        or where scaled_mad == 0 (constant window — cannot assess).
    is_outlier : pd.Series
        Boolean mask. True where abs(modified_z) > threshold.
    n_outliers : int
        Total number of flagged observations.
    outlier_dates : list
        Index values where is_outlier is True.
    window : int
        Rolling window used.
    threshold : float
        Modified z-score threshold used.
    outlier_rate : float
        n_outliers / total non-NaN observations. Useful for sanity checks.
    """
    modified_z: pd.Series
    is_outlier: pd.Series
    n_outliers: int
    outlier_dates: List
    window: int
    threshold: float
    outlier_rate: float = field(init=False)

    def __post_init__(self) -> None:
        n_assessed = int(self.is_outlier.notna().sum())
        self.outlier_rate = (
            self.n_outliers / n_assessed if n_assessed > 0 else 0.0
        )

    def summary(self) -> str:
        lines = [
            f"Window: {self.window}  Threshold: {self.threshold}",
            f"Outliers: {self.n_outliers}  Rate: {self.outlier_rate:.2%}",
        ]
        for date in self.outlier_dates[:10]:          # cap at 10 for readability
            mz = self.modified_z.loc[date]
            lines.append(f"  {pd.Timestamp(date).strftime('%Y-%m-%d')}  mod-z={mz:+.2f}")
        if len(self.outlier_dates) > 10:
            lines.append(f"  ... and {len(self.outlier_dates) - 10} more")
        return "\n".join(lines)


# ---------------------------------------------------------------------------
# Low-level primitives
# ---------------------------------------------------------------------------

def compute_mad(series: pd.Series) -> float:
    """
    Compute the Median Absolute Deviation of a series.

    MAD = median( |x - median(x)| )

    Parameters
    ----------
    series : pd.Series
        Any numeric series. NaNs are ignored.

    Returns
    -------
    float
        MAD value. Returns 0.0 if the series has fewer than 2 non-NaN values.
    """
    clean = series.dropna()
    if len(clean) < 2:
        return 0.0
    return float((clean - clean.median()).abs().median())


def global_modified_z_scores(series: pd.Series) -> pd.Series:
    """
    Compute modified z-scores using global (non-rolling) MAD.

    Use this for exploratory analysis and method comparisons.
    The pipeline uses detect_outliers_mad (rolling) instead.

    Parameters
    ----------
    series : pd.Series
        Numeric series (e.g. log returns).

    Returns
    -------
    pd.Series
        Modified z-scores. Returns all-zero Series if MAD == 0
        (constant input — no meaningful score possible).
    """
    median_val = series.median()
    mad = compute_mad(series)
    scaled_mad = MAD_CONSISTENCY_FACTOR * mad

    if scaled_mad == 0:
        logger.warning(
            "global_modified_z_scores: scaled_mad == 0 "
            "(constant or near-constant series). Returning zeros."
        )
        return pd.Series(0.0, index=series.index)

    return (series - median_val) / scaled_mad


# ---------------------------------------------------------------------------
# Rolling outlier detection  (used by the pipeline)
# ---------------------------------------------------------------------------

def detect_outliers_mad(
    returns: pd.Series,
    window: int = OUTLIER_WINDOW,
    threshold: float = OUTLIER_THRESHOLD,
    min_periods: int = OUTLIER_MIN_PERIODS,
) -> OutlierResult:
    """
    Detect outliers using a rolling MAD-based modified z-score.

    For each date t, the rolling window [t-window+1 … t] is used to compute
    a local median and MAD. The modified z-score for t is:

        mod_z(t) = (r(t) - rolling_median(t)) / (1.4826 * rolling_MAD(t))

    A return is flagged as an outlier if |mod_z| > threshold.

    Observations are marked NaN (not flagged) when:
      - Fewer than min_periods non-NaN values exist in the window.
      - The rolling MAD is zero (constant window — assessment impossible).

    Parameters
    ----------
    returns : pd.Series
        Log return series with a DatetimeIndex. NaNs are tolerated.
    window : int
        Rolling window size in trading days. Default from config (~3 months).
    threshold : float
        Modified z-score threshold. Default from config (3.5).
    min_periods : int
        Minimum non-NaN observations in a window before scoring starts.

    Returns
    -------
    OutlierResult
        Structured result with modified z-scores, boolean mask, dates,
        and summary statistics.
    """
    if returns.empty:
        raise ValueError("returns Series is empty — nothing to detect.")

    if len(returns.dropna()) < min_periods:
        raise ValueError(
            f"returns has only {len(returns.dropna())} non-NaN values, "
            f"but min_periods={min_periods}. Cannot detect outliers."
        )

    # ── Rolling median ────────────────────────────────────────────────────
    rolling_median = returns.rolling(
        window=window, min_periods=min_periods
    ).median()

    # ── Rolling MAD ───────────────────────────────────────────────────────
    deviations = (returns - rolling_median).abs()
    rolling_mad = deviations.rolling(
        window=window, min_periods=min_periods
    ).median()

    scaled_mad = MAD_CONSISTENCY_FACTOR * rolling_mad

    # Where scaled_mad == 0, the window is constant — score is undefined.
    # Use NaN so those dates are never accidentally flagged as outliers.
    scaled_mad_safe = scaled_mad.replace(0.0, np.nan)

    # ── Modified z-scores ─────────────────────────────────────────────────
    mod_z = (returns - rolling_median) / scaled_mad_safe

    # ── Flag outliers ─────────────────────────────────────────────────────
    is_outlier = mod_z.abs() > threshold

    n_outliers = int(is_outlier.sum())
    outlier_dates = returns.index[is_outlier].tolist()

    result = OutlierResult(
        modified_z=mod_z,
        is_outlier=is_outlier,
        n_outliers=n_outliers,
        outlier_dates=outlier_dates,
        window=window,
        threshold=threshold,
    )

    logger.info(
        "detect_outliers_mad: %d outlier(s) detected  "
        "(window=%d, threshold=%.1f, rate=%.2f%%)",
        n_outliers, window, threshold, result.outlier_rate * 100,
    )

    if n_outliers > 0:
        logger.debug("Outlier summary:\n%s", result.summary())

    return result


# ---------------------------------------------------------------------------
# Method comparison utility  (exploratory — not used by pipeline)
# ---------------------------------------------------------------------------

def compare_methods(
    returns: pd.Series,
    z_threshold: float = 3.0,
    mad_threshold: float = OUTLIER_THRESHOLD,
    mad_window: int = OUTLIER_WINDOW,
) -> dict:
    """
    Compare global z-score vs rolling MAD outlier detection side by side.

    Useful for analysis notebooks and validation. Not called by the pipeline.

    Parameters
    ----------
    returns : pd.Series
        Log return series.
    z_threshold : float
        Standard z-score cutoff for the global method.
    mad_threshold : float
        Modified z-score cutoff for the rolling MAD method.
    mad_window : int
        Rolling window for the MAD method.

    Returns
    -------
    dict with keys:
        z_outliers      — boolean Series from global z-score
        mad_result      — OutlierResult from rolling MAD
        n_z             — count flagged by z-score only
        n_mad           — count flagged by MAD only
        n_both          — count flagged by both
        n_only_mad      — flagged by MAD but not z-score (MAD's extra sensitivity)
    """
    z_scores = (returns - returns.mean()) / returns.std()
    z_outliers = z_scores.abs() > z_threshold

    mad_result = detect_outliers_mad(
        returns, window=mad_window, threshold=mad_threshold
    )
    mad_outliers = mad_result.is_outlier

    n_both     = int((z_outliers & mad_outliers).sum())
    n_only_mad = int((~z_outliers & mad_outliers).sum())

    logger.info(
        "compare_methods — z-score: %d  MAD: %d  both: %d  only-MAD: %d",
        int(z_outliers.sum()), mad_result.n_outliers, n_both, n_only_mad,
    )

    return {
        "z_outliers":  z_outliers,
        "mad_result":  mad_result,
        "n_z":         int(z_outliers.sum()),
        "n_mad":       mad_result.n_outliers,
        "n_both":      n_both,
        "n_only_mad":  n_only_mad,
    }