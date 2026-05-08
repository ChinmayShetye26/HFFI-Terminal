"""
Market data fetcher with flexible provider + universe selection.

Providers (set MARKET_PROVIDER in .env):
    - yfinance: free, slow at scale, good up to ~1000 tickers
    - polygon:  paid, fast, full US market real-time
    - alpaca:   paid free-tier, fast, full US market

Universes (set TICKER_UNIVERSE in .env):
    - sp500       : ~500 tickers (free, fast, recommended default)
    - russell1000 : ~1000 tickers (free with yfinance, slower)
    - full_us     : ~6000+ tickers (REQUIRES paid Polygon or Alpaca key)
    - custom      : reads tickers from data/custom_universe.txt

NETWORK NOTE: Live HTTP. Will not work in sandboxes without network.
"""

from __future__ import annotations
from datetime import datetime, timedelta
from pathlib import Path
from typing import Dict, List, Optional
from contextlib import contextmanager, redirect_stderr, redirect_stdout
import os
import logging
import time
import hashlib
import re
import io
import pandas as pd

logger = logging.getLogger(__name__)

TICKER_RE = re.compile(r"^[A-Z0-9.^=_-]{1,24}$")
SNAPSHOT_COLUMNS = ["price", "prev_close", "change", "change_pct", "volume", "market_cap", "name", "data_source"]
YFINANCE_COOLDOWN_MINUTES = int(os.getenv("YFINANCE_COOLDOWN_MINUTES", "30"))
YFINANCE_RATE_LIMIT_MARKERS = (
    "Too Many Requests",
    "YFRateLimitError",
    "Rate limited",
    "429",
)


def _sanitize_ticker(ticker: object) -> str:
    text = str(ticker or "").strip().upper().replace(" ", "")
    return text if TICKER_RE.fullmatch(text) else ""


def _yfinance_cooldown_path() -> Path:
    return Path(".cache/yfinance_rate_limited_until.txt")


def _looks_rate_limited(text: object) -> bool:
    haystack = str(text or "")
    return any(marker.lower() in haystack.lower() for marker in YFINANCE_RATE_LIMIT_MARKERS)


def _mark_yfinance_rate_limited(minutes: int = YFINANCE_COOLDOWN_MINUTES) -> None:
    path = _yfinance_cooldown_path()
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(str(time.time() + max(1, minutes) * 60))


def _is_yfinance_in_cooldown() -> bool:
    path = _yfinance_cooldown_path()
    if not path.exists():
        return False
    try:
        until = float(path.read_text().strip())
    except ValueError:
        return False
    if until <= time.time():
        return False
    return True


def _extract_yfinance_errors(yf_module: object) -> str:
    shared = getattr(yf_module, "shared", None)
    errors = getattr(shared, "_ERRORS", {}) if shared is not None else {}
    if not isinstance(errors, dict):
        return ""
    return "\n".join(f"{ticker}: {error}" for ticker, error in errors.items())


@contextmanager
def _quiet_yfinance():
    """Suppress yfinance console noise while still letting us inspect output."""
    captured = io.StringIO()
    loggers = [logging.getLogger(name) for name in ("yfinance", "yfinance.scrapers.history")]
    previous = [(log, log.level, log.propagate) for log in loggers]
    try:
        for log in loggers:
            log.setLevel(logging.CRITICAL)
            log.propagate = False
        with redirect_stdout(captured), redirect_stderr(captured):
            yield captured
    finally:
        for log, level, propagate in previous:
            log.setLevel(level)
            log.propagate = propagate


# --------------------------------------------------------------------------- #
# Universe construction
# --------------------------------------------------------------------------- #
def fetch_ticker_universe(universe: Optional[str] = None) -> List[str]:
    """Return a list of tickers based on the requested universe."""
    universe = (universe or os.getenv("TICKER_UNIVERSE", "sp500")).lower()

    if universe == "sp500":
        return _sp500_tickers()
    elif universe == "russell1000":
        return _russell1000_tickers()
    elif universe == "full_us":
        return _full_us_tickers()
    elif universe == "custom":
        custom_path = Path("data/custom_universe.txt")
        if not custom_path.exists():
            raise FileNotFoundError(
                f"TICKER_UNIVERSE=custom but {custom_path} not found. "
                "Create it with one ticker per line."
            )
        return [line.strip() for line in custom_path.read_text().splitlines() if line.strip()]
    else:
        raise ValueError(f"Unknown universe: {universe!r}")


def _sp500_tickers() -> List[str]:
    """Scrape current S&P 500 constituents from Wikipedia. Cached."""
    cache_path = Path(".cache/sp500_tickers.txt")
    if cache_path.exists() and (time.time() - cache_path.stat().st_mtime) < 86400:
        return [line.strip() for line in cache_path.read_text().splitlines() if line.strip()]

    try:
        tables = pd.read_html("https://en.wikipedia.org/wiki/List_of_S%26P_500_companies")
        tickers = tables[0]["Symbol"].astype(str).str.replace(".", "-", regex=False).tolist()
    except Exception as e:
        logger.warning("Could not scrape S&P 500 from Wikipedia: %s", e)
        # Hardcoded fallback (top 50 by market cap as of 2024 — replace as needed)
        tickers = [
            "AAPL", "MSFT", "NVDA", "GOOGL", "AMZN", "META", "TSLA", "BRK-B",
            "LLY", "AVGO", "JPM", "V", "WMT", "XOM", "UNH", "MA", "PG", "COST",
            "JNJ", "HD", "ORCL", "MRK", "ABBV", "BAC", "CVX", "KO", "ADBE",
            "NFLX", "PEP", "TMO", "CRM", "ACN", "AMD", "LIN", "MCD", "ABT",
            "WFC", "DHR", "CSCO", "TXN", "DIS", "VZ", "QCOM", "PM", "IBM",
            "AMGN", "INTU", "INTC", "CAT", "GS",
        ]
    cache_path.parent.mkdir(parents=True, exist_ok=True)
    cache_path.write_text("\n".join(tickers))
    return tickers


def _russell1000_tickers() -> List[str]:
    """Russell 1000 constituents — start with S&P 500 and add others.

    For a true Russell 1000 list you'd need to subscribe to FTSE Russell or
    use an iShares ETF holdings file (IWB). This stub returns S&P 500 plus
    a hand-picked set of mid-caps as a starting point.
    """
    base = _sp500_tickers()
    extras = [
        "PLTR", "RIVN", "COIN", "ROKU", "DOCU", "SNAP", "U", "DASH", "ABNB",
        "SQ", "SHOP", "SOFI", "AFRM", "HOOD", "DKNG", "PINS", "Z", "OPEN",
    ]
    return list(dict.fromkeys(base + extras))


def _full_us_tickers() -> List[str]:
    """Full US market — requires Polygon or Alpaca."""
    provider = os.getenv("MARKET_PROVIDER", "yfinance").lower()
    if provider == "polygon":
        return _polygon_full_us_tickers()
    elif provider == "alpaca":
        return _alpaca_full_us_tickers()
    else:
        raise RuntimeError(
            "TICKER_UNIVERSE=full_us requires MARKET_PROVIDER=polygon or alpaca. "
            "yfinance cannot enumerate the full US market reliably; Wikipedia "
            "doesn't host the full list. Either get a paid key, or fall back "
            "to TICKER_UNIVERSE=russell1000."
        )


def _polygon_full_us_tickers() -> List[str]:
    """Polygon.io: enumerate all active US common stocks."""
    try:
        from polygon import RESTClient
    except ImportError as e:
        raise ImportError("Run: pip install polygon-api-client") from e
    api_key = os.getenv("POLYGON_API_KEY")
    if not api_key:
        raise RuntimeError("POLYGON_API_KEY not set in .env")
    client = RESTClient(api_key)
    tickers = []
    for t in client.list_tickers(market="stocks", active=True, type="CS", limit=1000):
        tickers.append(t.ticker)
    return tickers


def _alpaca_full_us_tickers() -> List[str]:
    """Alpaca: enumerate all tradable US assets."""
    try:
        from alpaca.trading.client import TradingClient
    except ImportError as e:
        raise ImportError("Run: pip install alpaca-py") from e
    key, secret = os.getenv("ALPACA_API_KEY"), os.getenv("ALPACA_SECRET_KEY")
    if not (key and secret):
        raise RuntimeError("ALPACA_API_KEY and ALPACA_SECRET_KEY required")
    client = TradingClient(key, secret, paper=True)
    return [a.symbol for a in client.get_all_assets() if a.tradable]


# --------------------------------------------------------------------------- #
# Snapshot fetching
# --------------------------------------------------------------------------- #
def fetch_market_snapshot(
    tickers: Optional[List[str]] = None,
    provider: Optional[str] = None,
    use_cache: bool = True,
    cache_minutes: int = 5,
) -> pd.DataFrame:
    """Fetch current price + day change for the requested tickers.

    Returns a DataFrame indexed by ticker with columns:
        price, prev_close, change, change_pct, volume, market_cap, name
    """
    provider = (provider or os.getenv("MARKET_PROVIDER", "yfinance")).lower()
    tickers = tickers or fetch_ticker_universe()
    tickers = list(dict.fromkeys(t for t in (_sanitize_ticker(t) for t in tickers) if t))
    if not tickers:
        return _empty_market_frame()

    cache_key = hashlib.sha256("\n".join(sorted(tickers)).encode("utf-8")).hexdigest()[:12]
    cache_path = Path(f".cache/market_{provider}_{cache_key}.parquet")
    if use_cache and cache_path.exists():
        mtime = datetime.fromtimestamp(cache_path.stat().st_mtime)
        if datetime.now() - mtime < timedelta(minutes=cache_minutes):
            logger.info("Using cached market snapshot")
            return pd.read_parquet(cache_path)

    try:
        if provider == "yfinance":
            if _is_yfinance_in_cooldown():
                df = _fallback_snapshot(tickers, reason="yfinance_cooldown")
            else:
                df = _yfinance_snapshot(tickers)
        elif provider == "polygon":
            df = _polygon_snapshot(tickers)
        elif provider == "alpaca":
            df = _alpaca_snapshot(tickers)
        else:
            raise ValueError(f"Unknown provider: {provider!r}")
    except Exception as e:
        if provider == "yfinance" and _looks_rate_limited(e):
            _mark_yfinance_rate_limited()
        logger.info("Market provider %s failed; using offline fallback: %s", provider, e)
        df = _fallback_snapshot(tickers, reason=type(e).__name__)

    if df.empty:
        df = _fallback_snapshot(tickers, reason="empty_or_rate_limited")

    cache_path.parent.mkdir(parents=True, exist_ok=True)
    df.to_parquet(cache_path)
    return df


def _empty_market_frame() -> pd.DataFrame:
    return pd.DataFrame(columns=SNAPSHOT_COLUMNS).rename_axis("ticker")


def _fallback_snapshot(tickers: List[str], reason: str = "offline") -> pd.DataFrame:
    """Deterministic non-trading snapshot used when live providers rate-limit.

    Values are indicative placeholders so the UI and recommendation pipeline can
    continue to run. They are clearly marked with data_source='fallback:*'.
    """
    rows = []
    for ticker in tickers:
        seed = int(hashlib.sha256(ticker.encode("utf-8")).hexdigest()[:8], 16)
        base = 20.0 + (seed % 25000) / 100.0
        change_pct = ((seed % 1201) - 600) / 100000.0
        prev = base / (1.0 + change_pct) if abs(1.0 + change_pct) > 1e-9 else base
        rows.append({
            "ticker": ticker,
            "price": round(base, 2),
            "prev_close": round(prev, 2),
            "change": round(base - prev, 2),
            "change_pct": change_pct,
            "volume": float("nan"),
            "market_cap": float("nan"),
            "name": ticker,
            "data_source": f"fallback:{reason}",
        })
    if not rows:
        return _empty_market_frame()
    return pd.DataFrame(rows).set_index("ticker")


def _yfinance_snapshot(tickers: List[str]) -> pd.DataFrame:
    """yfinance snapshot. Slow above ~500 tickers — use S&P 500 as default."""
    try:
        import yfinance as yf
    except ImportError as e:
        raise ImportError("Run: pip install yfinance") from e

    if _is_yfinance_in_cooldown():
        return _empty_market_frame()

    rows = []
    try:
        # Batch download is faster, but threads=False is gentler on Yahoo's free endpoint.
        with _quiet_yfinance() as yf_output:
            data = yf.download(
                tickers, period="2d", interval="1d",
                group_by="ticker", auto_adjust=False, progress=False, threads=False,
            )
        diagnostic_text = f"{yf_output.getvalue()}\n{_extract_yfinance_errors(yf)}"
    except Exception as e:
        if _looks_rate_limited(e):
            _mark_yfinance_rate_limited()
            return _empty_market_frame()
        raise

    if _looks_rate_limited(diagnostic_text):
        _mark_yfinance_rate_limited()
        return _empty_market_frame()

    for ticker in tickers:
        try:
            if len(tickers) == 1:
                d = data
            else:
                d = data[ticker].dropna()
            if len(d) < 1:
                continue
            price = float(d["Close"].iloc[-1])
            prev = float(d["Close"].iloc[-2]) if len(d) >= 2 else float(d["Open"].iloc[-1])
            volume = float(d["Volume"].iloc[-1]) if "Volume" in d else float("nan")
            rows.append({
                "ticker": ticker,
                "price": price,
                "prev_close": prev,
                "change": price - prev,
                "change_pct": (price - prev) / prev if prev else float("nan"),
                "volume": volume,
            })
        except Exception as e:
            logger.debug("Skipping %s: %s", ticker, e)

    if not rows:
        return _empty_market_frame()
    df = pd.DataFrame(rows).set_index("ticker")
    df["market_cap"] = float("nan")
    df["name"] = df.index
    df["data_source"] = "live:yfinance"
    return df.reindex(columns=SNAPSHOT_COLUMNS)


def _polygon_snapshot(tickers: List[str]) -> pd.DataFrame:
    """Polygon.io snapshot. Fast and complete for full-US universes."""
    try:
        from polygon import RESTClient
    except ImportError as e:
        raise ImportError("Run: pip install polygon-api-client") from e
    api_key = os.getenv("POLYGON_API_KEY")
    if not api_key:
        raise RuntimeError("POLYGON_API_KEY not set")
    client = RESTClient(api_key)
    rows = []
    for snap in client.get_snapshot_all("stocks"):
        if snap.ticker not in tickers:
            continue
        rows.append({
            "ticker": snap.ticker,
            "price": snap.last_trade.price if snap.last_trade else float("nan"),
            "prev_close": snap.prev_day.close if snap.prev_day else float("nan"),
            "change": snap.todays_change,
            "change_pct": snap.todays_change_percent / 100,
            "volume": snap.day.volume if snap.day else float("nan"),
        })
    if not rows:
        return _empty_market_frame()
    df = pd.DataFrame(rows).set_index("ticker")
    df["market_cap"] = float("nan")
    df["name"] = df.index
    df["data_source"] = "live:polygon"
    return df.reindex(columns=SNAPSHOT_COLUMNS)


def _alpaca_snapshot(tickers: List[str]) -> pd.DataFrame:
    """Alpaca snapshot."""
    try:
        from alpaca.data.historical import StockHistoricalDataClient
        from alpaca.data.requests import StockSnapshotRequest
    except ImportError as e:
        raise ImportError("Run: pip install alpaca-py") from e
    key, secret = os.getenv("ALPACA_API_KEY"), os.getenv("ALPACA_SECRET_KEY")
    client = StockHistoricalDataClient(key, secret)
    snaps = client.get_stock_snapshot(StockSnapshotRequest(symbol_or_symbols=tickers))
    rows = []
    for ticker, snap in snaps.items():
        try:
            price = snap.latest_trade.price
            prev = snap.previous_daily_bar.close
            rows.append({
                "ticker": ticker,
                "price": price, "prev_close": prev,
                "change": price - prev,
                "change_pct": (price - prev) / prev if prev else float("nan"),
                "volume": snap.daily_bar.volume if snap.daily_bar else float("nan"),
            })
        except Exception as e:
            logger.debug("Skipping %s: %s", ticker, e)
    if not rows:
        return _empty_market_frame()
    df = pd.DataFrame(rows).set_index("ticker")
    df["market_cap"] = float("nan")
    df["name"] = df.index
    df["data_source"] = "live:alpaca"
    return df.reindex(columns=SNAPSHOT_COLUMNS)
