# market_pipeline/gaps.py
"""
Gap detection and forward-fill for OHLCV DataFrames.

What this module does
---------------------
1. detect_gaps        — find stretches where consecutive dates are too far apart
2. reindex_to_calendar — expand the DataFrame to every business day in range,
                         inserting NaN rows for missing dates
3. forward_fill_prices — fill NaN prices (ffill) and NaN volume (zero-fill)
4. fill_gaps           — convenience orchestrator: reindex then fill in one call

A note on market calendars
--------------------------
By default this module uses pd.bdate_range (Mon–Fri, no holidays).
This means exchange holidays (Good Friday, Diwali, Thanksgiving, etc.) will
appear as missing business days even though markets were correctly closed.

For production use with a specific exchange, install pandas_market_calendars
and pass a calendar name:

    fill_gaps(df, calendar='XNSE')     # NSE India
    fill_gaps(df, calendar='XNYS')     # NYSE

If pandas_market_calendars is not installed and a calendar name is passed,
a clear ImportError is raised — not a silent fallback.
"""

import importlib
import logging
from dataclasses import dataclass, field
from typing import List, Optional

import numpy as np
import pandas as pd

from market_pipeline.config import GAP_THRESHOLD_DAYS

logger = logging.getLogger(__name__)

_PRICE_COLS = ["Open", "High", "Low", "Close"]


# ---------------------------------------------------------------------------
# Return types
# ---------------------------------------------------------------------------

@dataclass
class GapInfo:
    """
    A single detected gap between two consecutive trading dates.

    Attributes
    ----------
    start : pd.Timestamp
        Last date with data before the gap.
    end : pd.Timestamp
        First date with data after the gap.
    calendar_days : int
        Total calendar days between start and end.
    business_days_missing : int
        Estimated business days missing (Mon–Fri, ignoring holidays).
    """
    start: pd.Timestamp
    end: pd.Timestamp
    calendar_days: int
    business_days_missing: int

    def __str__(self) -> str:
        return (
            f"{self.start.strftime('%Y-%m-%d')} → {self.end.strftime('%Y-%m-%d')}  "
            f"({self.calendar_days} calendar days, "
            f"{self.business_days_missing} business days missing)"
        )


@dataclass
class FillReport:
    """
    Summary of what forward_fill_prices filled.

    Attributes
    ----------
    filled_counts : dict
        Per-column count of NaN values that were filled.
    filled_dates : list
        Index values of every row that had at least one NaN filled.
    total_filled_cells : int
        Sum of all per-column fill counts.
    n_rows_filled : int
        Number of distinct rows that had any fill applied.
    """
    filled_counts: dict = field(default_factory=dict)
    filled_dates: List = field(default_factory=list)

    @property
    def total_filled_cells(self) -> int:
        return sum(
            v for v in self.filled_counts.values() if isinstance(v, int)
        )

    @property
    def n_rows_filled(self) -> int:
        return len(self.filled_dates)

    def summary(self) -> str:
        if self.total_filled_cells == 0:
            return "No fills applied — data was complete."
        lines = [
            f"Total cells filled : {self.total_filled_cells}",
            f"Rows affected      : {self.n_rows_filled}",
        ]
        for col, count in self.filled_counts.items():
            if count > 0:
                lines.append(f"  {col}: {count}")
        return "\n".join(lines)


# ---------------------------------------------------------------------------
# Internal helpers
# ---------------------------------------------------------------------------

def _build_trading_calendar(
    start: pd.Timestamp,
    end: pd.Timestamp,
    calendar: Optional[str],
) -> pd.DatetimeIndex:
    """
    Return a DatetimeIndex of valid trading days between start and end.

    Parameters
    ----------
    start, end : pd.Timestamp
        Inclusive date range.
    calendar : str or None
        pandas_market_calendars exchange name (e.g. 'XNYS', 'XNSE').
        If None, falls back to generic Mon–Fri business days.
    """
    if calendar is not None:
        try:
            mcal = importlib.import_module("pandas_market_calendars")
        except ImportError:
            raise ImportError(
                "pandas_market_calendars is required to use a named calendar. "
                "Install it with:  pip install pandas-market-calendars\n"
                "Or pass calendar=None to use generic Mon–Fri business days."
            )
        cal = mcal.get_calendar(calendar)
        sessions = cal.sessions_in_range(start, end)
        return pd.DatetimeIndex(sessions.normalize())

    # Generic fallback — Mon–Fri, no holiday awareness
    return pd.bdate_range(start, end)


# ---------------------------------------------------------------------------
# Core functions
# ---------------------------------------------------------------------------

def detect_gaps(
    dates: pd.DatetimeIndex,
    threshold: int = GAP_THRESHOLD_DAYS,
) -> List[GapInfo]:
    """
    Find stretches where consecutive dates are further apart than threshold.

    Parameters
    ----------
    dates : pd.DatetimeIndex
        Sorted trading dates from a DataFrame's index.
    threshold : int
        Minimum calendar-day gap to flag. Default from config (5 days,
        i.e. anything larger than a normal weekend).

    Returns
    -------
    list of GapInfo
        One entry per detected gap, sorted chronologically.

    Raises
    ------
    ValueError
        If dates has fewer than 2 entries (no consecutive pairs to compare).
    """
    if len(dates) < 2:
        raise ValueError(
            f"detect_gaps requires at least 2 dates, got {len(dates)}."
        )

    dates_series = pd.Series(dates)
    day_diffs = (dates_series - dates_series.shift(1)).dt.days.dropna()

    gaps: List[GapInfo] = []
    for i, diff in enumerate(day_diffs, start=1):
        if diff >= threshold:
            start = dates[i - 1]
            end   = dates[i]
            # Subtract 2 to exclude endpoints themselves
            bdays = max(0, len(pd.bdate_range(start, end)) - 2)
            gaps.append(GapInfo(
                start=start,
                end=end,
                calendar_days=int(diff),
                business_days_missing=bdays,
            ))

    logger.info(
        "detect_gaps: found %d gap(s) larger than %d calendar days",
        len(gaps), threshold,
    )
    for gap in gaps:
        logger.debug("  %s", gap)

    return gaps


def reindex_to_calendar(
    df: pd.DataFrame,
    calendar: Optional[str] = None,
) -> tuple[pd.DataFrame, pd.DatetimeIndex]:
    """
    Expand the DataFrame to every valid trading day in its date range.

    Rows for missing dates are inserted with NaN values. The caller should
    then pass the result to forward_fill_prices().

    Parameters
    ----------
    df : pd.DataFrame
        OHLCV DataFrame with a DatetimeIndex.
    calendar : str or None
        Exchange calendar name for pandas_market_calendars.
        None → generic Mon–Fri business days.

    Returns
    -------
    tuple of (pd.DataFrame, pd.DatetimeIndex)
        - Reindexed DataFrame (NaN rows for missing dates).
        - DatetimeIndex of the dates that were missing (the inserted rows).
    """
    if df.empty:
        raise ValueError("Cannot reindex an empty DataFrame.")

    full_index = _build_trading_calendar(df.index[0], df.index[-1], calendar)
    reindexed  = df.reindex(full_index)
    reindexed.index.name = "Date"

    missing_dates = full_index[reindexed["Close"].isna()]

    logger.info(
        "reindex_to_calendar: %d trading days in range, "
        "%d had data, %d missing (holidays / gaps)",
        len(full_index), len(df), len(missing_dates),
    )

    return reindexed, missing_dates


def forward_fill_prices(df: pd.DataFrame) -> tuple[pd.DataFrame, FillReport]:
    """
    Forward-fill missing prices; zero-fill missing volume.

    Fill strategy
    -------------
    - Open, High, Low, Close : forward-filled (last known price carried forward).
      This reflects market convention — the price does not change on a closed day.
    - Volume : filled with 0. A missing volume means no trades occurred.
      Using ffill for volume would make a zero-liquidity day look active,
      which corrupts backtests and risk models.

    Note: forward-filling Open is an approximation. On a real holiday the
    next day's Open is a fresh auction price, not the prior Close. This is
    a known limitation of calendar-blind gap filling.

    Parameters
    ----------
    df : pd.DataFrame
        OHLCV DataFrame, typically the output of reindex_to_calendar().

    Returns
    -------
    tuple of (pd.DataFrame, FillReport)
        Filled DataFrame and a structured fill report.
    """
    if df.empty:
        raise ValueError("Cannot fill an empty DataFrame.")

    result  = df.copy()
    report  = FillReport()

    # Track which rows had any NaN before filling
    any_nan_before = result.isna().any(axis=1)

    # ── Price columns — forward fill ──────────────────────────────────────
    for col in _PRICE_COLS:
        if col not in result.columns:
            continue
        n_missing = int(result[col].isna().sum())
        if n_missing > 0:
            report.filled_counts[col] = n_missing
        result[col] = result[col].ffill()

    # ── Volume — zero fill ────────────────────────────────────────────────
    if "Volume" in result.columns:
        n_missing = int(result["Volume"].isna().sum())
        if n_missing > 0:
            report.filled_counts["Volume"] = n_missing
        result["Volume"] = result["Volume"].fillna(0)

    # ── Record which rows were touched ────────────────────────────────────
    report.filled_dates = result.index[any_nan_before].tolist()

    logger.info("forward_fill_prices: %s", report.summary())
    return result, report


# ---------------------------------------------------------------------------
# Convenience orchestrator
# ---------------------------------------------------------------------------

def fill_gaps(
    df: pd.DataFrame,
    calendar: Optional[str] = None,
    gap_threshold: int = GAP_THRESHOLD_DAYS,
) -> tuple[pd.DataFrame, List[GapInfo], FillReport]:
    """
    Detect gaps, reindex to trading calendar, and forward-fill — in one call.

    This is the function pipeline.py calls. The individual functions above
    are available for inspection and testing.

    Parameters
    ----------
    df : pd.DataFrame
        Raw OHLCV DataFrame with DatetimeIndex.
    calendar : str or None
        Exchange calendar name. None → generic Mon–Fri.
    gap_threshold : int
        Minimum calendar days to flag as a named gap.

    Returns
    -------
    tuple of (pd.DataFrame, list[GapInfo], FillReport)
        - Filled DataFrame aligned to full trading calendar.
        - List of large named gaps found before reindexing.
        - FillReport describing every cell that was filled.
    """
    gaps      = detect_gaps(df.index, threshold=gap_threshold)
    reindexed, _ = reindex_to_calendar(df, calendar=calendar)
    filled, fill_report = forward_fill_prices(reindexed)

    return filled, gaps, fill_report