# market_pipeline/adjustments.py
"""
Corporate action adjustments for OHLCV data.

Handles two types of events:
  - Stock splits  : divide historical prices by ratio, multiply volume by ratio
  - Cash dividends: multiply historical prices by (1 - dividend / prior_close)

Both adjustments work backwards — they modify all rows *before* the event
date so the price series is continuous and return calculations are correct.

Design decisions
----------------
- Events are dataclasses, not raw dicts. A typo like 'ex_date' vs 'exdate'
  raises immediately at construction, not silently mid-pipeline.
- Multiple events are sorted chronologically before application. Order
  matters: applying a later split before an earlier one produces wrong factors.
- A detailed AdjustmentReport is returned alongside the data so callers
  know exactly what was applied and can audit the pipeline.
- Nothing here fetches events from a vendor. Providing the splits/dividends
  calendar is the caller's responsibility (pass it from your data vendor,
  a CSV, yfinance corporate actions, etc.).
"""

import logging
from dataclasses import dataclass, field
from typing import Dict, List, Optional

import pandas as pd

logger = logging.getLogger(__name__)

_PRICE_COLS = ["Open", "High", "Low", "Close"]


# ---------------------------------------------------------------------------
# Input types
# ---------------------------------------------------------------------------

@dataclass
class SplitEvent:
    """
    A single stock split.

    Parameters
    ----------
    date : str or pd.Timestamp
        The date the split took effect (ex-date).
        All rows strictly before this date will be adjusted.
    ratio : float
        Split ratio. A 4-for-1 split → ratio=4.
        Pre-split prices are divided by ratio; volume is multiplied.
    """
    date: str
    ratio: float

    def __post_init__(self) -> None:
        self.date = pd.Timestamp(self.date)
        if self.ratio <= 0:
            raise ValueError(
                f"SplitEvent ratio must be positive, got {self.ratio}."
            )
        if self.ratio == 1.0:
            logger.warning("SplitEvent ratio=1.0 on %s — this is a no-op.", self.date.date())


@dataclass
class DividendEvent:
    """
    A single cash dividend.

    Parameters
    ----------
    ex_date : str or pd.Timestamp
        The ex-dividend date. All rows strictly before this date are adjusted.
    amount : float
        Gross dividend amount in the same currency as the price series.
    """
    ex_date: str
    amount: float

    def __post_init__(self) -> None:
        self.ex_date = pd.Timestamp(self.ex_date)
        if self.amount <= 0:
            raise ValueError(
                f"DividendEvent amount must be positive, got {self.amount}."
            )


# ---------------------------------------------------------------------------
# Return type
# ---------------------------------------------------------------------------

@dataclass
class AdjustmentReport:
    """
    Audit trail of every corporate action that was applied.

    Attributes
    ----------
    splits_applied : list of dict
        One entry per applied split: date, ratio.
    splits_skipped : list of dict
        Splits skipped because they fell outside the data range.
    dividends_applied : list of dict
        One entry per applied dividend: ex_date, amount, factor, prior_close.
    dividends_skipped : list of dict
        Dividends skipped because of invalid factor or no prior data.
    """
    splits_applied: List[Dict] = field(default_factory=list)
    splits_skipped: List[Dict] = field(default_factory=list)
    dividends_applied: List[Dict] = field(default_factory=list)
    dividends_skipped: List[Dict] = field(default_factory=list)

    @property
    def n_splits(self) -> int:
        return len(self.splits_applied)

    @property
    def n_dividends(self) -> int:
        return len(self.dividends_applied)

    def summary(self) -> str:
        lines = [
            f"Splits   applied: {self.n_splits}  skipped: {len(self.splits_skipped)}",
            f"Dividends applied: {self.n_dividends}  skipped: {len(self.dividends_skipped)}",
        ]
        for s in self.splits_applied:
            lines.append(f"  Split  {s['date'].date()}  ratio={s['ratio']}")
        for d in self.dividends_applied:
            lines.append(
                f"  Div    {d['ex_date'].date()}  "
                f"amount={d['amount']:.4f}  factor={d['factor']:.6f}"
            )
        return "\n".join(lines)


# ---------------------------------------------------------------------------
# Core adjustment functions
# ---------------------------------------------------------------------------

def adjust_for_splits(
    df: pd.DataFrame,
    splits: List[SplitEvent],
) -> tuple[pd.DataFrame, AdjustmentReport]:
    """
    Back-adjust historical prices and volume for stock splits.

    For each split, all rows strictly before the split date are divided
    by the ratio (prices) or multiplied by the ratio (volume).
    Events are applied in chronological order.

    Parameters
    ----------
    df : pd.DataFrame
        OHLCV DataFrame with a DatetimeIndex.
    splits : list of SplitEvent
        Split events to apply.

    Returns
    -------
    tuple of (pd.DataFrame, AdjustmentReport)
        Adjusted DataFrame and a report of what was applied.
    """
    adjusted = df.copy()
    report = AdjustmentReport()

    if not splits:
        return adjusted, report

    # Always apply in chronological order
    for split in sorted(splits, key=lambda s: s.date):
        before = adjusted.index < split.date

        if not before.any():
            logger.warning(
                "Split on %s falls before all data — skipping.", split.date.date()
            )
            report.splits_skipped.append({"date": split.date, "ratio": split.ratio,
                                           "reason": "no data before split date"})
            continue

        for col in _PRICE_COLS:
            if col in adjusted.columns:
                adjusted.loc[before, col] = adjusted.loc[before, col] / split.ratio

        if "Volume" in adjusted.columns:
            adjusted.loc[before, "Volume"] = (
                adjusted.loc[before, "Volume"] * split.ratio
            )

        logger.info("Applied split: date=%s  ratio=%s", split.date.date(), split.ratio)
        report.splits_applied.append({"date": split.date, "ratio": split.ratio})

    return adjusted, report


def adjust_for_dividends(
    df: pd.DataFrame,
    dividends: List[DividendEvent],
) -> tuple[pd.DataFrame, AdjustmentReport]:
    """
    Back-adjust historical prices for cash dividends.

    For each dividend, compute an adjustment factor:
        factor = 1 - (amount / prior_close)

    All rows strictly before the ex-date are multiplied by this factor.
    Events are applied in chronological order.

    Parameters
    ----------
    df : pd.DataFrame
        OHLCV DataFrame with a DatetimeIndex.
    dividends : list of DividendEvent
        Dividend events to apply.

    Returns
    -------
    tuple of (pd.DataFrame, AdjustmentReport)
        Adjusted DataFrame and a report of what was applied.
    """
    adjusted = df.copy()
    report = AdjustmentReport()

    if not dividends:
        return adjusted, report

    for div in sorted(dividends, key=lambda d: d.ex_date):
        before = adjusted.index < div.ex_date

        if not before.any():
            logger.warning(
                "Dividend ex-date %s has no prior data — skipping.", div.ex_date.date()
            )
            report.dividends_skipped.append({
                "ex_date": div.ex_date, "amount": div.amount,
                "reason": "no data before ex-date",
            })
            continue

        prior_close = adjusted.loc[before, "Close"].iloc[-1]

        if prior_close <= 0:
            logger.warning(
                "Prior close is non-positive (%.4f) for dividend on %s — skipping.",
                prior_close, div.ex_date.date(),
            )
            report.dividends_skipped.append({
                "ex_date": div.ex_date, "amount": div.amount,
                "reason": f"prior_close={prior_close:.4f} is non-positive",
            })
            continue

        factor = 1.0 - div.amount / prior_close

        # Guard: factor must produce a valid price series
        if not (0.0 < factor < 1.0):
            logger.warning(
                "Dividend factor %.6f is invalid for ex-date %s "
                "(amount=%.4f, prior_close=%.4f) — skipping.",
                factor, div.ex_date.date(), div.amount, prior_close,
            )
            report.dividends_skipped.append({
                "ex_date": div.ex_date, "amount": div.amount,
                "reason": f"factor={factor:.6f} out of (0, 1)",
            })
            continue

        for col in _PRICE_COLS:
            if col in adjusted.columns:
                adjusted.loc[before, col] = adjusted.loc[before, col] * factor

        logger.info(
            "Applied dividend: ex_date=%s  amount=%.4f  factor=%.6f",
            div.ex_date.date(), div.amount, factor,
        )
        report.dividends_applied.append({
            "ex_date": div.ex_date,
            "amount": div.amount,
            "factor": factor,
            "prior_close": prior_close,
        })

    return adjusted, report


# ---------------------------------------------------------------------------
# Convenience orchestrator
# ---------------------------------------------------------------------------

def apply_corporate_actions(
    df: pd.DataFrame,
    splits: Optional[List[SplitEvent]] = None,
    dividends: Optional[List[DividendEvent]] = None,
) -> tuple[pd.DataFrame, AdjustmentReport]:
    """
    Apply splits and dividends in a single call.

    Splits are applied first, then dividends. This matches the standard
    convention: price continuity is established for splits before the
    smaller dividend adjustments are layered on top.

    Parameters
    ----------
    df : pd.DataFrame
        Raw OHLCV DataFrame.
    splits : list of SplitEvent, optional
    dividends : list of DividendEvent, optional

    Returns
    -------
    tuple of (pd.DataFrame, AdjustmentReport)
        Fully adjusted DataFrame and a combined AdjustmentReport.
    """
    combined_report = AdjustmentReport()
    working = df.copy()

    if splits:
        working, split_report = adjust_for_splits(working, splits)
        combined_report.splits_applied  = split_report.splits_applied
        combined_report.splits_skipped  = split_report.splits_skipped

    if dividends:
        working, div_report = adjust_for_dividends(working, dividends)
        combined_report.dividends_applied = div_report.dividends_applied
        combined_report.dividends_skipped = div_report.dividends_skipped

    logger.info("Corporate actions complete.\n%s", combined_report.summary())
    return working, combined_report