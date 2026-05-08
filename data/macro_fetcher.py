"""
FRED macro data fetcher.

Pulls the live macro indicators needed for the macro-fragility component
and for the terminal's "macro tape" panel:

    - Inflation rate (CPI YoY)         — CPIAUCSL
    - Unemployment rate                 — UNRATE
    - Federal funds rate                — FEDFUNDS
    - 30-year mortgage rate             — MORTGAGE30US
    - 10-year Treasury yield            — DGS10
    - 2-year Treasury yield             — DGS2
    - VIX (market stress proxy)         — VIXCLS
    - Real GDP growth (QoQ annualized)  — A191RL1Q225SBEA

Free FRED API key required — sign up at:
    https://fred.stlouisfed.org/docs/api/api_key.html

Set FRED_API_KEY in your .env file.

NETWORK NOTE: This module makes live HTTP requests. It will not work in
sandboxed environments without network access. Run it on your own machine
or in a notebook where the FRED API is reachable.
"""

from __future__ import annotations
from datetime import datetime, timedelta
from pathlib import Path
from typing import Dict, Optional
import os
import logging
import pandas as pd

logger = logging.getLogger(__name__)


# All series IDs in one place for easy maintenance.
FRED_SERIES = {
    "cpi":              "CPIAUCSL",
    "unemployment":     "UNRATE",
    "fed_funds":        "FEDFUNDS",
    "mortgage_30y":     "MORTGAGE30US",
    "treasury_10y":     "DGS10",
    "treasury_2y":      "DGS2",
    "vix":              "VIXCLS",
    "real_gdp_growth":  "A191RL1Q225SBEA",
}


def _get_fred_client():
    """Lazy import + connect when a FRED API key is configured.

    The dashboard can still fetch macro data without a key by using FRED's
    public chart CSV endpoint. Returning None here keeps missing keys from
    surfacing as a RuntimeError in Streamlit.
    """
    api_key = os.getenv("FRED_API_KEY")
    if not api_key:
        return None
    try:
        from fredapi import Fred
    except ImportError as e:
        raise ImportError(
            "fredapi not installed. Run: pip install fredapi"
        ) from e
    return Fred(api_key=api_key)


def _fetch_public_fred_series(series_id: str, start: Optional[str] = None) -> pd.Series:
    """Fetch a FRED series without an API key via the public chart CSV."""
    url = f"https://fred.stlouisfed.org/graph/fredgraph.csv?id={series_id}"
    df = pd.read_csv(url)
    if df.empty or len(df.columns) < 2:
        return pd.Series(dtype=float)
    date_col = df.columns[0]
    value_col = series_id if series_id in df.columns else df.columns[-1]
    dates = pd.to_datetime(df[date_col], errors="coerce")
    values = pd.to_numeric(df[value_col], errors="coerce")
    series = pd.Series(values.to_numpy(), index=dates).dropna()
    if start:
        series = series[series.index >= pd.to_datetime(start)]
    return series


def _fetch_series(series_id: str, start: Optional[str] = None) -> pd.Series:
    """Fetch a FRED series using fredapi when possible, public CSV otherwise."""
    fred = _get_fred_client()
    if fred is not None:
        kwargs = {"observation_start": start} if start else {}
        return fred.get_series(series_id, **kwargs).dropna()
    return _fetch_public_fred_series(series_id, start=start)


def fetch_macro_snapshot(use_cache: bool = True, cache_minutes: int = 60) -> Dict[str, float]:
    """Pull the latest available value for each macro series.

    Returns a dict with keys matching FRED_SERIES plus derived fields:
        inflation_rate    — YoY % change in CPI, as decimal (0.034 = 3.4%)
        fed_funds_rate    — current FFR as decimal
        unemployment_rate — current UR as decimal
        mortgage_rate     — current 30y mortgage as decimal
        treasury_10y, treasury_2y, yield_curve_spread — all as decimal
        vix               — raw points (not decimal)
        timestamp         — ISO datetime of fetch

    All "rate" fields are returned as decimals (0.05 not 5.0) for direct
    use in the macro_fragility component.
    """
    cache_path = Path(".cache/macro_snapshot.parquet")
    if use_cache and cache_path.exists():
        mtime = datetime.fromtimestamp(cache_path.stat().st_mtime)
        if datetime.now() - mtime < timedelta(minutes=cache_minutes):
            logger.info("Using cached macro snapshot")
            df = pd.read_parquet(cache_path)
            return df.iloc[0].to_dict()

    snapshot = {}

    # Pull each series; FRED returns a pandas Series indexed by date.
    for name, sid in FRED_SERIES.items():
        try:
            s = _fetch_series(sid)
            if len(s) == 0:
                snapshot[name] = float("nan")
                continue
            snapshot[f"{name}_latest"] = float(s.iloc[-1])
            snapshot[f"{name}_date"] = str(s.index[-1].date())
        except Exception as e:
            logger.warning("Failed to fetch %s (%s): %s", name, sid, e)
            snapshot[f"{name}_latest"] = float("nan")

    # CPI is reported as an index level — compute YoY inflation.
    try:
        cpi_series = _fetch_series(FRED_SERIES["cpi"])
        if len(cpi_series) >= 13:
            yoy = (cpi_series.iloc[-1] / cpi_series.iloc[-13]) - 1.0
            snapshot["inflation_rate"] = float(yoy)
        else:
            snapshot["inflation_rate"] = float("nan")
    except Exception as e:
        logger.warning("CPI YoY computation failed: %s", e)
        snapshot["inflation_rate"] = float("nan")

    # Convenience aliases as decimals (rates are reported as percent in FRED)
    snapshot["unemployment_rate"] = snapshot.get("unemployment_latest", float("nan")) / 100.0
    snapshot["fed_funds_rate"]    = snapshot.get("fed_funds_latest", float("nan")) / 100.0
    snapshot["mortgage_rate"]     = snapshot.get("mortgage_30y_latest", float("nan")) / 100.0
    snapshot["treasury_10y"]      = snapshot.get("treasury_10y_latest", float("nan")) / 100.0
    snapshot["treasury_2y"]       = snapshot.get("treasury_2y_latest", float("nan")) / 100.0
    snapshot["yield_curve_spread"] = snapshot["treasury_10y"] - snapshot["treasury_2y"]
    snapshot["vix"]                = snapshot.get("vix_latest", float("nan"))
    snapshot["real_gdp_growth"]    = snapshot.get("real_gdp_growth_latest", float("nan")) / 100.0
    snapshot["timestamp"] = datetime.utcnow().isoformat()

    # Cache for next call
    cache_path.parent.mkdir(parents=True, exist_ok=True)
    pd.DataFrame([snapshot]).to_parquet(cache_path)

    return snapshot


def fetch_macro_history(
    series: str = "fed_funds",
    start: str = "2000-01-01",
) -> pd.Series:
    """Fetch historical time series for plotting in the terminal."""
    sid = FRED_SERIES.get(series, series)
    return _fetch_series(sid, start=start)
