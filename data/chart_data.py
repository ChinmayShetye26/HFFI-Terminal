"""
Live chart data fetcher.

Provides OHLC history for any ticker, with provider flexibility:
    - yfinance: free, daily + intraday for ~7 days back
    - polygon: paid, intraday tick-level history
    - alpaca: paid, similar to polygon

Returns DataFrames in a uniform format that the Streamlit UI can plot
directly with Plotly's candlestick + line chart components.
"""

from __future__ import annotations
from datetime import datetime, timedelta
from pathlib import Path
from typing import Optional
import os
import logging
import re
import hashlib
import numpy as np
import pandas as pd

from data.market_fetcher import (
    _extract_yfinance_errors,
    _is_yfinance_in_cooldown,
    _looks_rate_limited,
    _mark_yfinance_rate_limited,
    _quiet_yfinance,
)

logger = logging.getLogger(__name__)

TICKER_RE = re.compile(r"^[A-Z0-9.^=_-]{1,24}$")


def _sanitize_ticker(ticker: object) -> str:
    text = str(ticker or "").strip().upper().replace(" ", "")
    if not TICKER_RE.fullmatch(text):
        raise ValueError("Ticker contains unsupported characters.")
    return text


def fetch_history(
    ticker: str,
    period: str = "1y",
    interval: str = "1d",
    provider: Optional[str] = None,
) -> pd.DataFrame:
    """Fetch OHLC history for a single ticker.

    Parameters
    ----------
    ticker : symbol (use the yfinance form: 'AAPL', 'GLD', 'EURUSD=X', '^GSPC')
    period : '1d' | '5d' | '1mo' | '3mo' | '6mo' | '1y' | '2y' | '5y' | '10y' | 'ytd' | 'max'
    interval : '1m' | '5m' | '15m' | '30m' | '60m' | '1h' | '1d' | '1wk' | '1mo'

    Returns
    -------
    DataFrame indexed by timestamp with columns: Open, High, Low, Close, Volume
    """
    ticker = _sanitize_ticker(ticker)
    provider = (provider or os.getenv("MARKET_PROVIDER", "yfinance")).lower()

    if provider == "yfinance":
        if _is_yfinance_in_cooldown():
            return _fallback_history(ticker, period, interval, reason="yfinance_cooldown")
        try:
            return _yfinance_history(ticker, period, interval)
        except Exception as e:
            if _looks_rate_limited(e):
                _mark_yfinance_rate_limited()
            logger.info("yfinance history failed; using offline fallback: %s", e)
            return _fallback_history(ticker, period, interval, reason=type(e).__name__)
    if provider == "polygon":
        try:
            return _polygon_history(ticker, period, interval)
        except Exception as e:
            logger.warning("Polygon history failed (%s); falling back to yfinance", e)
            return _fallback_history(ticker, period, interval, reason=type(e).__name__)
    if provider == "alpaca":
        try:
            return _alpaca_history(ticker, period, interval)
        except Exception as e:
            logger.warning("Alpaca history failed (%s); falling back to yfinance", e)
            return _fallback_history(ticker, period, interval, reason=type(e).__name__)
    return _fallback_history(ticker, period, interval, reason="unknown_provider")


def _yfinance_history(ticker: str, period: str, interval: str) -> pd.DataFrame:
    try:
        import yfinance as yf
    except ImportError as e:
        raise ImportError("Run: pip install yfinance") from e

    if _is_yfinance_in_cooldown():
        return _fallback_history(ticker, period, interval, reason="yfinance_cooldown")

    try:
        with _quiet_yfinance() as yf_output:
            t = yf.Ticker(ticker)
            df = t.history(period=period, interval=interval, auto_adjust=False)
        diagnostic_text = f"{yf_output.getvalue()}\n{_extract_yfinance_errors(yf)}"
    except Exception as e:
        if _looks_rate_limited(e):
            _mark_yfinance_rate_limited()
            return _fallback_history(ticker, period, interval, reason="rate_limited")
        raise

    if _looks_rate_limited(diagnostic_text):
        _mark_yfinance_rate_limited()
        return _fallback_history(ticker, period, interval, reason="rate_limited")
    if df.empty:
        return _fallback_history(ticker, period, interval, reason="empty_or_rate_limited")
    # Standardize column names
    df = df.rename(columns={"Open": "Open", "High": "High", "Low": "Low",
                            "Close": "Close", "Volume": "Volume"})
    return df[["Open", "High", "Low", "Close", "Volume"]]


def _fallback_history(ticker: str, period: str, interval: str, reason: str = "offline") -> pd.DataFrame:
    """Deterministic OHLC fallback for rate-limited/no-network demos."""
    periods = {
        "1d": 24, "5d": 60, "1mo": 30, "3mo": 65, "6mo": 130,
        "1y": 252, "2y": 504, "5y": 756, "10y": 1008, "max": 1260,
    }
    n = periods.get(period, 252)
    freq = "h" if interval in {"1m", "5m", "15m", "30m", "60m", "1h"} and period in {"1d", "5d"} else "B"
    idx = pd.date_range(end=pd.Timestamp.utcnow().tz_localize(None), periods=n, freq=freq)
    seed = int(hashlib.sha256(ticker.encode("utf-8")).hexdigest()[:8], 16)
    rng = np.random.default_rng(seed)
    start = 20.0 + (seed % 25000) / 100.0
    rets = rng.normal(0.0003, 0.012, n)
    close = start * np.cumprod(1 + rets)
    spread = np.maximum(close * 0.008, 0.01)
    open_ = np.r_[close[0], close[:-1]]
    high = np.maximum(open_, close) + spread * rng.uniform(0.2, 1.0, n)
    low = np.minimum(open_, close) - spread * rng.uniform(0.2, 1.0, n)
    volume = rng.integers(100_000, 5_000_000, n)
    df = pd.DataFrame({
        "Open": open_,
        "High": high,
        "Low": low,
        "Close": close,
        "Volume": volume,
    }, index=idx)
    df.attrs["data_source"] = f"fallback:{reason}"
    return df


def _polygon_history(ticker: str, period: str, interval: str) -> pd.DataFrame:
    try:
        from polygon import RESTClient
    except ImportError as e:
        raise ImportError("Run: pip install polygon-api-client") from e
    api_key = os.getenv("POLYGON_API_KEY")
    if not api_key:
        raise RuntimeError("POLYGON_API_KEY not set")
    client = RESTClient(api_key)

    period_to_days = {"1d": 1, "5d": 5, "1mo": 30, "3mo": 90, "6mo": 180,
                      "1y": 365, "2y": 730, "5y": 1825, "10y": 3650, "max": 7300}
    days = period_to_days.get(period, 365)
    end = datetime.utcnow().date()
    start = end - timedelta(days=days)

    interval_map = {"1m": (1, "minute"), "5m": (5, "minute"), "15m": (15, "minute"),
                    "30m": (30, "minute"), "60m": (1, "hour"), "1h": (1, "hour"),
                    "1d": (1, "day"), "1wk": (1, "week"), "1mo": (1, "month")}
    multiplier, span = interval_map.get(interval, (1, "day"))

    aggs = client.get_aggs(ticker, multiplier, span, start, end, limit=50000)
    rows = []
    for a in aggs:
        rows.append({
            "Open": a.open, "High": a.high, "Low": a.low,
            "Close": a.close, "Volume": a.volume,
            "timestamp": pd.to_datetime(a.timestamp, unit="ms"),
        })
    df = pd.DataFrame(rows).set_index("timestamp")
    return df


def _alpaca_history(ticker: str, period: str, interval: str) -> pd.DataFrame:
    try:
        from alpaca.data.historical import StockHistoricalDataClient
        from alpaca.data.requests import StockBarsRequest
        from alpaca.data.timeframe import TimeFrame, TimeFrameUnit
    except ImportError as e:
        raise ImportError("Run: pip install alpaca-py") from e
    key = os.getenv("ALPACA_API_KEY"); secret = os.getenv("ALPACA_SECRET_KEY")
    client = StockHistoricalDataClient(key, secret)

    period_to_days = {"1d": 1, "5d": 5, "1mo": 30, "3mo": 90, "6mo": 180,
                      "1y": 365, "2y": 730, "5y": 1825, "10y": 3650}
    days = period_to_days.get(period, 365)
    start = datetime.utcnow() - timedelta(days=days)

    interval_map = {"1m": TimeFrame(1, TimeFrameUnit.Minute), "5m": TimeFrame(5, TimeFrameUnit.Minute),
                    "15m": TimeFrame(15, TimeFrameUnit.Minute), "1h": TimeFrame(1, TimeFrameUnit.Hour),
                    "1d": TimeFrame(1, TimeFrameUnit.Day)}
    tf = interval_map.get(interval, TimeFrame(1, TimeFrameUnit.Day))

    req = StockBarsRequest(symbol_or_symbols=ticker, timeframe=tf, start=start)
    bars = client.get_stock_bars(req).df
    if bars.empty:
        return bars
    bars = bars.reset_index()
    bars = bars.rename(columns={"open": "Open", "high": "High", "low": "Low",
                                 "close": "Close", "volume": "Volume",
                                 "timestamp": "timestamp"})
    return bars.set_index("timestamp")[["Open", "High", "Low", "Close", "Volume"]]
