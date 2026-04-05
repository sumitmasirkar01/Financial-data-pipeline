"""Market data pipeline package."""
# market_pipeline/__init__.py
"""
market_pipeline — financial OHLCV data quality and engineering pipeline.

Quick start
-----------
    from market_pipeline.pipeline import run_pipeline_for_tickers

    results = run_pipeline_for_tickers(['AAPL', 'MSFT', 'RELIANCE.NS'])
    for ticker, result in results.items():
        print(ticker, result.quality.grade, result.quality.quality_score)
"""

__version__ = "1.0.0"