# market_pipeline/config.py
"""
Central configuration for the market data pipeline.

All hardcoded values live here. Change something once, it
updates everywhere. Never edit values inside the modules themselves.
"""

# ---------------------------------------------------------------------------
# Data fetching
# ---------------------------------------------------------------------------

# Default tickers to run when no explicit list is provided
DEFAULT_TICKERS: list[str] = ["AAPL", "MSFT", "TSLA", "GOOG"]

# Indian (NSE) tickers
INDIAN_TICKERS: list[str] = [
    "RELIANCE.NS",
    "TCS.NS",
    "HDFCBANK.NS",
    "INFY.NS",
]

# yfinance period string — how far back to pull data
# Valid values: "6mo", "1y", "2y", "5y", "max"
DEFAULT_PERIOD: str = "2y"

# ---------------------------------------------------------------------------
# Outlier detection  (MAD-based rolling z-score)
# ---------------------------------------------------------------------------

# Rolling window size in trading days (~3 months)
OUTLIER_WINDOW: int = 63

# Modified z-score threshold above which a return is flagged as an outlier
OUTLIER_THRESHOLD: float = 3.5

# Minimum observations required before the rolling calculation kicks in
OUTLIER_MIN_PERIODS: int = 20

# Consistency factor that makes MAD a consistent estimator of std dev
# under normality. Do not change unless you know what you are doing.
MAD_CONSISTENCY_FACTOR: float = 1.4826

# ---------------------------------------------------------------------------
# Gap detection
# ---------------------------------------------------------------------------

# Minimum calendar-day gap between two consecutive dates to be flagged
GAP_THRESHOLD_DAYS: int = 5

# ---------------------------------------------------------------------------
# Returns and volatility
# ---------------------------------------------------------------------------

# Trading days in a year — used to annualise volatility and returns
TRADING_DAYS_PER_YEAR: int = 252

# Rolling windows (in trading days) for realised volatility
VOL_WINDOW_SHORT: int = 20   # ~1 month
VOL_WINDOW_LONG: int = 60    # ~3 months

# ---------------------------------------------------------------------------
# Value at Risk (VaR)
# ---------------------------------------------------------------------------

# Default confidence level for VaR calculation
VAR_CONFIDENCE_LEVEL: float = 0.99

# Default notional principal (in local currency) for cash VaR
VAR_DEFAULT_PRINCIPAL: float = 100_000.0

# ---------------------------------------------------------------------------
# Quality scoring — grade boundaries
# ---------------------------------------------------------------------------
# quality_score is a float in [0, 1].  Score >= boundary earns that grade.

GRADE_BOUNDARIES: dict[str, float] = {
    "A": 0.99,
    "B": 0.95,
    "C": 0.90,
    "D": 0.80,
    # anything below D boundary → "F"
}

# Sanity cap: if the outlier rate exceeds this, flag it as suspicious
MAX_EXPECTED_OUTLIER_RATE: float = 0.05   # 5 %

# ---------------------------------------------------------------------------
# Visualisation — colour palette (dark professional theme)
# ---------------------------------------------------------------------------

COLORS: dict[str, str] = {
    "primary":   "#26804a",   # Phthalo green
    "secondary": "#339b5e",
    "accent":    "#56b67d",
    "light":     "#8dd1a8",
    "red":       "#f85149",
    "orange":    "#f0883e",
    "blue":      "#58a6ff",
    "purple":    "#bc8cff",
    "gray":      "#8b949e",
    "white":     "#e6edf3",
}

PLOT_STYLE: str = "dark_background"

PLOT_RC: dict = {
    "figure.figsize":      (14, 6),
    "figure.dpi":          150,
    "axes.grid":           True,
    "grid.alpha":          0.3,
    "axes.spines.top":     False,
    "axes.spines.right":   False,
    "font.size":           11,
}