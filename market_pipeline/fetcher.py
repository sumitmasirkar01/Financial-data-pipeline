# market_pipeline/fetcher.py
"""
Data fetching layer for OHLCV market data.

Responsibilities
----------------
- Pull daily OHLCV data from yfinance
- Retry on transient network / throttle failures
- Validate that the returned DataFrame is usable
- Isolate per-ticker failures when fetching many tickers at once

Nothing in this module touches outliers, gaps, or quality scoring.
Those live in their own modules.
"""

import logging
import time
import warnings
from typing import Dict, Optional

import pandas as pd
import yfinance as yf

from market_pipeline.config import DEFAULT_PERIOD

# Suppress yfinance's own noisy warnings — not everything else
warnings.filterwarnings("ignore", category=FutureWarning, module="yfinance")
warnings.filterwarnings("ignore", category=UserWarning, module="yfinance")

# Module-level logger — inherits the root logger's handler and level,
# so whoever runs your pipeline controls verbosity via logging.basicConfig()
logger = logging.getLogger(__name__)

# Columns we expect yfinance to return
_REQUIRED_COLUMNS = {"Open", "High", "Low", "Close", "Volume"}

# Valid period strings yfinance accepts
VALID_PERIODS = {"1d", "5d", "1mo", "3mo", "6mo", "1y", "2y", "5y", "10y", "ytd", "max"}


# ---------------------------------------------------------------------------
# Internal helpers
# ---------------------------------------------------------------------------

def _validate_period(period: str) -> None:
    """Raise ValueError if period string is not recognised by yfinance."""
    if period not in VALID_PERIODS:
        raise ValueError(
            f"Invalid period '{period}'. "
            f"Must be one of: {sorted(VALID_PERIODS)}"
        )


def _clean_index(df: pd.DataFrame) -> pd.DataFrame:
    """Strip timezone from DatetimeIndex and name it 'Date'."""
    df.index = pd.to_datetime(df.index.date)
    df.index.name = "Date"
    return df


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------

def fetch_ohlcv(
    ticker: str,
    period: str = DEFAULT_PERIOD,
    retries: int = 3,
    backoff: float = 2.0,
) -> pd.DataFrame:
    """
    Fetch daily OHLCV data for a single ticker.

    Parameters
    ----------
    ticker : str
        Stock symbol (e.g. 'AAPL', 'RELIANCE.NS').
    period : str
        Look-back window. One of: 1d 5d 1mo 3mo 6mo 1y 2y 5y 10y ytd max.
    retries : int
        How many times to retry on transient failures before raising.
    backoff : float
        Seconds to wait before the first retry. Doubles on each attempt.

    Returns
    -------
    pd.DataFrame
        DataFrame with DatetimeIndex and columns:
        Open, High, Low, Close, Volume  — in that order.

    Raises
    ------
    ValueError
        If the ticker is empty, the period is invalid, or yfinance returns
        no data (ticker probably doesn't exist or is delisted).
    RuntimeError
        If all retry attempts are exhausted due to network errors.
    """
    # ── Input guards ──────────────────────────────────────────────────────
    if not ticker or not isinstance(ticker, str):
        raise ValueError("ticker must be a non-empty string.")
    ticker = ticker.strip().upper()
    _validate_period(period)

    # ── Fetch with retry ──────────────────────────────────────────────────
    last_exc: Optional[Exception] = None
    wait = backoff

    for attempt in range(1, retries + 1):
        try:
            logger.debug("Fetching %s (period=%s, attempt=%d)", ticker, period, attempt)
            stock = yf.Ticker(ticker)
            raw = stock.history(period=period, auto_adjust=False)
            break                          # success — exit the retry loop

        except Exception as exc:           # yfinance raises broad exceptions
            last_exc = exc
            if attempt < retries:
                logger.warning(
                    "Fetch failed for %s (attempt %d/%d): %s — retrying in %.1fs",
                    ticker, attempt, retries, exc, wait,
                )
                time.sleep(wait)
                wait *= 2                  # exponential backoff
            else:
                raise RuntimeError(
                    f"All {retries} fetch attempts failed for '{ticker}'. "
                    f"Last error: {last_exc}"
                ) from last_exc

    # ── Validate returned data ────────────────────────────────────────────
    if raw is None or raw.empty:
        raise ValueError(
            f"yfinance returned no data for '{ticker}' with period='{period}'. "
            "The ticker may be invalid, delisted, or not yet traded."
        )

    missing_cols = _REQUIRED_COLUMNS - set(raw.columns)
    if missing_cols:
        raise ValueError(
            f"yfinance response for '{ticker}' is missing columns: {missing_cols}. "
            f"Got: {list(raw.columns)}"
        )

    # ── Select and clean ─────────────────────────────────────────────────
    df = raw[["Open", "High", "Low", "Close", "Volume"]].copy()
    df = _clean_index(df)

    # The Feynman Shield: Drop rows where yfinance returns a "phantom" day of fully missing data
    df = df.dropna(subset=["Open", "High", "Low", "Close"], how="all")

    # If the dataframe became empty after dropping phantom days
    if df.empty:
        raise ValueError(f"yfinance returned only empty/phantom data for '{ticker}'.")

    logger.info(
        "Fetched %s: %d rows  (%s → %s)",
        ticker,
        len(df),
        df.index[0].strftime("%Y-%m-%d"),
        df.index[-1].strftime("%Y-%m-%d"),
    )
    return df


def fetch_multiple(
    tickers: list[str],
    period: str = DEFAULT_PERIOD,
    retries: int = 3,
    backoff: float = 2.0,
) -> Dict[str, pd.DataFrame]:
    """
    Fetch OHLCV data for a list of tickers.

    Unlike calling fetch_ohlcv in a plain loop, this function isolates
    failures — one bad ticker does not abort the rest.

    Parameters
    ----------
    tickers : list[str]
        List of stock symbols.
    period : str
        Look-back window (same options as fetch_ohlcv).
    retries : int
        Retry attempts per ticker.
    backoff : float
        Initial backoff seconds (doubles on each retry).

    Returns
    -------
    dict
        Keys are ticker symbols. Values are DataFrames for successful
        fetches. Failed tickers are logged as errors and excluded.
    """
    if not tickers:
        raise ValueError("tickers list is empty.")

    results: Dict[str, pd.DataFrame] = {}
    failed: list[str] = []

    for ticker in tickers:
        try:
            results[ticker] = fetch_ohlcv(ticker, period=period,
                                           retries=retries, backoff=backoff)
        except (ValueError, RuntimeError) as exc:
            logger.error("Skipping %s — %s", ticker, exc)
            failed.append(ticker)

    if failed:
        logger.warning(
            "%d ticker(s) could not be fetched and were skipped: %s",
            len(failed), failed,
        )

    if not results:
        raise RuntimeError(
            "No data could be fetched for any of the provided tickers. "
            f"All failed: {tickers}"
        )

    return results