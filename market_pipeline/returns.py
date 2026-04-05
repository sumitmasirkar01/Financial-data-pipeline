# market_pipeline/returns.py
"""
Return computation, summary statistics, and Value at Risk.

Functions
---------
compute_returns       — adds return and volatility columns to an OHLCV DataFrame
compute_max_drawdown  — max peak-to-trough drawdown from a return series
compute_return_stats  — annualised return, vol, Sharpe, skew, kurtosis, max DD
calculate_var         — Historical and Parametric Value at Risk

VaR sign convention
-------------------
VaR is a *loss* measure. All values in VaRResult are stored as
positive numbers representing the magnitude of potential loss.
  historical_var_pct = 0.032  →  3.2% potential loss
  historical_var_cash = 3200  →  ₹3,200 / $3,200 potential loss

This is the convention used by risk desks. The original notebook stored
negative floats and called abs() at print time — that approach breaks
silently when a caller forgets to negate.
"""

import logging
from dataclasses import dataclass

import numpy as np
import pandas as pd
from scipy import stats

from market_pipeline.config import (
    TRADING_DAYS_PER_YEAR,
    VAR_CONFIDENCE_LEVEL,
    VAR_DEFAULT_PRINCIPAL,
    VOL_WINDOW_LONG,
    VOL_WINDOW_SHORT,
)

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Return types
# ---------------------------------------------------------------------------

@dataclass
class ReturnStats:
    """
    Annualised summary statistics for a return series.

    All volatility and return figures are annualised.
    Skewness and kurtosis are computed on daily log returns.

    Attributes
    ----------
    ann_return : float      Annualised mean log return.
    ann_vol    : float      Annualised realised volatility (std of log returns).
    sharpe     : float      ann_return / ann_vol. 0.0 if vol == 0.
    skewness   : float      Skewness of daily log returns. Normal = 0.
    kurtosis   : float      Excess kurtosis. Normal = 0. Fat tails > 0.
    max_drawdown : float    Maximum peak-to-trough drawdown (positive = loss).
    n_obs      : int        Number of non-NaN return observations.
    """
    ann_return:   float
    ann_vol:      float
    sharpe:       float
    skewness:     float
    kurtosis:     float
    max_drawdown: float
    n_obs:        int

    def summary(self) -> str:
        return (
            f"Ann. return   : {self.ann_return:.2%}\n"
            f"Ann. vol      : {self.ann_vol:.2%}\n"
            f"Sharpe        : {self.sharpe:.2f}\n"
            f"Skewness      : {self.skewness:.3f}\n"
            f"Excess kurtosis: {self.kurtosis:.2f}  "
            f"({'fat tails' if self.kurtosis > 1 else 'near-normal'})\n"
            f"Max drawdown  : {self.max_drawdown:.2%}\n"
            f"Observations  : {self.n_obs}"
        )


@dataclass
class VaRResult:
    """
    Value at Risk output. All loss figures are positive.

    Attributes
    ----------
    confidence_level     : float  e.g. 0.99 for 99% VaR.
    historical_var_pct   : float  Loss % from empirical distribution.
    parametric_var_pct   : float  Loss % assuming normal distribution.
    historical_var_cash  : float  historical_var_pct × principal (positive).
    parametric_var_cash  : float  parametric_var_pct × principal (positive).
    principal            : float  Notional investment.
    n_obs                : int    Observations used.
    """
    confidence_level:    float
    historical_var_pct:  float
    parametric_var_pct:  float
    historical_var_cash: float
    parametric_var_cash: float
    principal:           float
    n_obs:               int

    def summary(self) -> str:
        return (
            f"VaR @ {self.confidence_level:.0%} confidence\n"
            f"  Historical : {self.historical_var_pct:.2%}  "
            f"(cash: {self.historical_var_cash:,.2f})\n"
            f"  Parametric : {self.parametric_var_pct:.2%}  "
            f"(cash: {self.parametric_var_cash:,.2f})\n"
            f"  Principal  : {self.principal:,.0f}  "
            f"  Obs: {self.n_obs}"
        )


# ---------------------------------------------------------------------------
# Core functions
# ---------------------------------------------------------------------------

def compute_returns(df: pd.DataFrame) -> pd.DataFrame:
    """
    Add return and rolling volatility columns to an OHLCV DataFrame.

    Added columns
    -------------
    simple_return     — (Close_t / Close_{t-1}) - 1
    log_return        — ln(Close_t / Close_{t-1})
    realized_vol_20d  — 20-day rolling annualised vol (from config)
    realized_vol_60d  — 60-day rolling annualised vol (from config)

    Parameters
    ----------
    df : pd.DataFrame
        Must contain a 'Close' column.

    Returns
    -------
    pd.DataFrame
        Original DataFrame with four new columns appended.

    Raises
    ------
    ValueError
        If 'Close' column is missing or has fewer than 2 non-NaN values.
    """
    if "Close" not in df.columns:
        raise ValueError("DataFrame must contain a 'Close' column.")

    close = df["Close"]
    n_valid = int(close.notna().sum())
    if n_valid < 2:
        raise ValueError(
            f"'Close' has only {n_valid} non-NaN value(s). "
            "Need at least 2 to compute returns."
        )

    result = df.copy()

    # Calculate the ratio first
    ratio = close / close.shift(1)
    
    # The Feynman Shield: Force anything <= 0 to become NaN. 
    # np.log will safely ignore NaNs, producing another NaN.
    safe_ratio = ratio.where(ratio > 0, np.nan)

    result["simple_return"] = close.pct_change()
    result["log_return"]    = np.log(safe_ratio)

    ann_factor = np.sqrt(TRADING_DAYS_PER_YEAR)

    result["realized_vol_20d"] = (
        result["log_return"]
        .rolling(window=VOL_WINDOW_SHORT)
        .std() * ann_factor
    )
    result["realized_vol_60d"] = (
        result["log_return"]
        .rolling(window=VOL_WINDOW_LONG)
        .std() * ann_factor
    )

    logger.debug(
        "compute_returns: added simple_return, log_return, "
        "realized_vol_%dd, realized_vol_%dd",
        VOL_WINDOW_SHORT, VOL_WINDOW_LONG,
    )
    return result


def compute_max_drawdown(simple_returns: pd.Series) -> float:
    """
    Compute the maximum peak-to-trough drawdown.

    Returns a positive float representing the largest percentage loss
    from any peak to the subsequent trough.

    Parameters
    ----------
    simple_returns : pd.Series
        Daily simple (not log) return series.

    Returns
    -------
    float
        Maximum drawdown as a positive decimal (e.g. 0.34 = 34% drawdown).
        Returns 0.0 if the series has no valid data.
    """
    # Force inf to NaN, then fill NaNs with 0.0 so the cumulative product ignores them
    clean = simple_returns.replace([np.inf, -np.inf], np.nan).fillna(0.0)
    
    if clean.empty:
        return 0.0

    cum_ret     = (1.0 + clean).cumprod()
    running_max = cum_ret.cummax()
    drawdown    = (cum_ret - running_max) / running_max

    return float(abs(drawdown.min()))


def compute_return_stats(
    log_returns: pd.Series,
    simple_returns: pd.Series,
) -> ReturnStats:
    """
    Compute annualised summary statistics for a return series.

    Parameters
    ----------
    log_returns : pd.Series
        Daily log return series. Used for return, vol, Sharpe, skew, kurtosis.
    simple_returns : pd.Series
        Daily simple return series. Used for max drawdown calculation only.

    Returns
    -------
    ReturnStats
        Dataclass with all summary statistics.

    Raises
    ------
    ValueError
        If log_returns has fewer than 2 non-NaN observations.
    """
    # Force inf and -inf to NaN before dropping
    clean = log_returns.replace([np.inf, -np.inf], np.nan).dropna()
    
    if len(clean) < 2:
        raise ValueError(
            f"Need at least 2 non-NaN log returns, got {len(clean)}."
        )

    ann_return = float(clean.mean() * TRADING_DAYS_PER_YEAR)
    ann_vol    = float(clean.std() * np.sqrt(TRADING_DAYS_PER_YEAR))
    sharpe     = ann_return / ann_vol if ann_vol > 0 else 0.0
    max_dd     = compute_max_drawdown(simple_returns)

    return ReturnStats(
        ann_return=ann_return,
        ann_vol=ann_vol,
        sharpe=sharpe,
        skewness=float(clean.skew()),
        kurtosis=float(clean.kurtosis()),
        max_drawdown=max_dd,
        n_obs=len(clean),
    )


def calculate_var(
    returns: pd.Series,
    confidence_level: float = VAR_CONFIDENCE_LEVEL,
    principal: float = VAR_DEFAULT_PRINCIPAL,
) -> VaRResult:
    """
    Calculate Historical and Parametric Value at Risk.

    Historical VaR
    --------------
    Reads directly from the empirical return distribution. Makes no
    assumption about the distribution shape. More accurate for fat-tailed
    assets like equities — it will typically show a larger loss than
    Parametric VaR precisely because it captures the real tail behaviour.

    Parametric VaR
    --------------
    Assumes returns are normally distributed. Uses mean and std of the
    observed series to compute the loss at the given quantile via the
    inverse normal CDF. Understates risk for fat-tailed assets.

    Parameters
    ----------
    returns : pd.Series
        Daily log return series.
    confidence_level : float
        Probability that actual loss will not exceed VaR.
        e.g. 0.99 → "we are 99% confident losses won't exceed this."
    principal : float
        Notional investment in local currency.

    Returns
    -------
    VaRResult
        All loss figures are positive. See class docstring for convention.

    Raises
    ------
    ValueError
        If confidence_level is not in (0, 1), or returns is empty.
    """
    if not (0.0 < confidence_level < 1.0):
        raise ValueError(
            f"confidence_level must be in (0, 1), got {confidence_level}."
        )

    # Force inf and -inf to NaN before dropping
    clean = returns.replace([np.inf, -np.inf], np.nan).dropna()
    
    if len(clean) < 2:
        raise ValueError(
            f"Need at least 2 non-NaN returns for VaR, got {len(clean)}."
        )

    # ── Historical VaR ────────────────────────────────────────────────────
    # percentile at the loss tail — already negative for typical return dists
    hist_pct_raw = float(np.percentile(clean, (1.0 - confidence_level) * 100))

    # ── Parametric VaR ────────────────────────────────────────────────────
    mu    = float(clean.mean())
    sigma = float(clean.std())
    z     = float(stats.norm.ppf(1.0 - confidence_level))   # negative value
    param_pct_raw = mu + z * sigma

    # ── Enforce positive-loss convention ─────────────────────────────────
    # Raw percentile values are negative (a return of -3% = -0.03).
    # We store the absolute value so callers never need to negate.
    hist_var_pct  = abs(hist_pct_raw)
    param_var_pct = abs(param_pct_raw)

    result = VaRResult(
        confidence_level=confidence_level,
        historical_var_pct=hist_var_pct,
        parametric_var_pct=param_var_pct,
        historical_var_cash=hist_var_pct * principal,
        parametric_var_cash=param_var_pct * principal,
        principal=principal,
        n_obs=len(clean),
    )

    logger.info("calculate_var:\n%s", result.summary())
    return result