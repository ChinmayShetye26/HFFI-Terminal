"""
Asset universe with full categorization.

Provides every category needed for the markets dashboard:
    - US equities (S&P 500, full US market)
    - Sector ETFs
    - Commodities (oil, gold, silver, etc.)
    - Forex (major pairs)
    - Bonds / treasuries
    - Crypto (optional, off by default)
    - Major indices

Each asset is tagged with category + subcategory so the UI can group them
into red/green tiles and the recommender can use sector exposure as a
feature.
"""

from __future__ import annotations
from dataclasses import dataclass
from typing import Dict, List, Optional
import os
from pathlib import Path
import logging

logger = logging.getLogger(__name__)


@dataclass
class Asset:
    ticker: str
    name: str
    category: str       # equity | commodity | forex | bond | sector | index | crypto
    subcategory: str    # e.g. 'tech', 'energy', 'precious_metals', 'major_pair'
    yfinance_symbol: Optional[str] = None  # if different from ticker

    def fetch_symbol(self) -> str:
        return self.yfinance_symbol or self.ticker


# --------------------------------------------------------------------------- #
# Sector ETFs (SPDR sector funds — the standard for sector tracking)
# --------------------------------------------------------------------------- #
SECTOR_ETFS = [
    Asset("XLK", "Technology",            "sector", "technology"),
    Asset("XLF", "Financials",             "sector", "financials"),
    Asset("XLV", "Healthcare",             "sector", "healthcare"),
    Asset("XLE", "Energy",                 "sector", "energy"),
    Asset("XLI", "Industrials",            "sector", "industrials"),
    Asset("XLY", "Consumer Discretionary", "sector", "consumer_discretionary"),
    Asset("XLP", "Consumer Staples",       "sector", "consumer_staples"),
    Asset("XLU", "Utilities",              "sector", "utilities"),
    Asset("XLRE", "Real Estate",           "sector", "real_estate"),
    Asset("XLB", "Materials",              "sector", "materials"),
    Asset("XLC", "Communications",         "sector", "communications"),
]

# --------------------------------------------------------------------------- #
# Commodities (futures-based ETFs and direct commodity tickers)
# --------------------------------------------------------------------------- #
COMMODITIES = [
    Asset("GLD",  "Gold (SPDR)",          "commodity", "precious_metals"),
    Asset("SLV",  "Silver (iShares)",     "commodity", "precious_metals"),
    Asset("USO",  "Crude Oil (USO)",      "commodity", "energy"),
    Asset("UNG",  "Natural Gas (UNG)",    "commodity", "energy"),
    Asset("DBA",  "Agriculture (DBA)",    "commodity", "agriculture"),
    Asset("CORN", "Corn (CORN)",          "commodity", "agriculture"),
    Asset("WEAT", "Wheat (WEAT)",         "commodity", "agriculture"),
    Asset("DBC",  "Broad Commodities",    "commodity", "broad"),
    Asset("CPER", "Copper (CPER)",        "commodity", "industrial_metals"),
    Asset("PALL", "Palladium",            "commodity", "precious_metals"),
    Asset("PPLT", "Platinum",             "commodity", "precious_metals"),
]

# --------------------------------------------------------------------------- #
# Forex (yfinance uses =X suffix for pairs)
# --------------------------------------------------------------------------- #
FOREX = [
    Asset("EURUSD", "EUR/USD", "forex", "major_pair", yfinance_symbol="EURUSD=X"),
    Asset("GBPUSD", "GBP/USD", "forex", "major_pair", yfinance_symbol="GBPUSD=X"),
    Asset("USDJPY", "USD/JPY", "forex", "major_pair", yfinance_symbol="JPY=X"),
    Asset("USDCHF", "USD/CHF", "forex", "major_pair", yfinance_symbol="CHF=X"),
    Asset("AUDUSD", "AUD/USD", "forex", "major_pair", yfinance_symbol="AUDUSD=X"),
    Asset("USDCAD", "USD/CAD", "forex", "major_pair", yfinance_symbol="CAD=X"),
    Asset("USDCNY", "USD/CNY", "forex", "emerging_pair", yfinance_symbol="CNY=X"),
    Asset("USDINR", "USD/INR", "forex", "emerging_pair", yfinance_symbol="INR=X"),
    Asset("USDMXN", "USD/MXN", "forex", "emerging_pair", yfinance_symbol="MXN=X"),
    Asset("DXY",    "USD Dollar Index", "forex", "index", yfinance_symbol="DX-Y.NYB"),
]

# --------------------------------------------------------------------------- #
# Bonds + treasuries
# --------------------------------------------------------------------------- #
BONDS = [
    Asset("AGG",  "US Aggregate Bonds",      "bond", "broad"),
    Asset("BND",  "Vanguard Total Bond",     "bond", "broad"),
    Asset("TLT",  "20+ Year Treasuries",      "bond", "long_treasury"),
    Asset("IEF",  "7-10 Year Treasuries",     "bond", "intermediate_treasury"),
    Asset("SHY",  "1-3 Year Treasuries",      "bond", "short_treasury"),
    Asset("LQD",  "Investment Grade Corp",    "bond", "corporate"),
    Asset("HYG",  "High Yield Corp",          "bond", "high_yield"),
    Asset("TIP",  "TIPS (inflation-linked)",  "bond", "tips"),
    Asset("MUB",  "Municipal Bonds",           "bond", "municipal"),
    Asset("EMB",  "Emerging Market Bonds",    "bond", "emerging_market"),
]

# --------------------------------------------------------------------------- #
# Major indices
# --------------------------------------------------------------------------- #
INDICES = [
    Asset("SPX",  "S&P 500",        "index", "us_large_cap", yfinance_symbol="^GSPC"),
    Asset("DJI",  "Dow Jones",      "index", "us_blue_chip", yfinance_symbol="^DJI"),
    Asset("IXIC", "Nasdaq Comp",    "index", "us_tech", yfinance_symbol="^IXIC"),
    Asset("RUT",  "Russell 2000",    "index", "us_small_cap", yfinance_symbol="^RUT"),
    Asset("VIX",  "Volatility Index","index", "volatility", yfinance_symbol="^VIX"),
    Asset("NIKKEI","Nikkei 225",     "index", "japan", yfinance_symbol="^N225"),
    Asset("FTSE", "FTSE 100",        "index", "uk", yfinance_symbol="^FTSE"),
    Asset("DAX",  "DAX",             "index", "germany", yfinance_symbol="^GDAXI"),
    Asset("HSI",  "Hang Seng",       "index", "hong_kong", yfinance_symbol="^HSI"),
]

# --------------------------------------------------------------------------- #
# Top US equities (curated mega-caps, plus full S&P 500 lookup)
# --------------------------------------------------------------------------- #
MEGA_CAP_EQUITIES = [
    # Tech
    Asset("AAPL",  "Apple",         "equity", "technology"),
    Asset("MSFT",  "Microsoft",     "equity", "technology"),
    Asset("GOOGL", "Alphabet",      "equity", "technology"),
    Asset("AMZN",  "Amazon",        "equity", "consumer_discretionary"),
    Asset("META",  "Meta",          "equity", "communications"),
    Asset("NVDA",  "NVIDIA",        "equity", "technology"),
    Asset("TSLA",  "Tesla",         "equity", "consumer_discretionary"),
    # Financials
    Asset("BRK-B", "Berkshire",     "equity", "financials"),
    Asset("JPM",   "JPMorgan",      "equity", "financials"),
    Asset("V",     "Visa",          "equity", "financials"),
    Asset("MA",    "Mastercard",    "equity", "financials"),
    Asset("BAC",   "Bank of America","equity", "financials"),
    # Healthcare
    Asset("UNH",   "UnitedHealth",   "equity", "healthcare"),
    Asset("JNJ",   "Johnson & Johnson","equity", "healthcare"),
    Asset("LLY",   "Eli Lilly",      "equity", "healthcare"),
    Asset("PFE",   "Pfizer",         "equity", "healthcare"),
    # Energy
    Asset("XOM",   "ExxonMobil",     "equity", "energy"),
    Asset("CVX",   "Chevron",        "equity", "energy"),
    # Consumer
    Asset("WMT",   "Walmart",        "equity", "consumer_staples"),
    Asset("PG",    "Procter & Gamble","equity", "consumer_staples"),
    Asset("KO",    "Coca-Cola",      "equity", "consumer_staples"),
    Asset("PEP",   "PepsiCo",        "equity", "consumer_staples"),
    Asset("HD",    "Home Depot",     "equity", "consumer_discretionary"),
    Asset("MCD",   "McDonald's",     "equity", "consumer_discretionary"),
    Asset("NKE",   "Nike",           "equity", "consumer_discretionary"),
    # Industrials
    Asset("CAT",   "Caterpillar",    "equity", "industrials"),
    Asset("BA",    "Boeing",         "equity", "industrials"),
    Asset("GE",    "GE Aerospace",   "equity", "industrials"),
]

# --------------------------------------------------------------------------- #
# Composite registry
# --------------------------------------------------------------------------- #
def build_full_registry() -> List[Asset]:
    """Return the union of all curated assets."""
    return (
        SECTOR_ETFS + COMMODITIES + FOREX + BONDS + INDICES + MEGA_CAP_EQUITIES
    )


def get_assets_by_category(category: str) -> List[Asset]:
    """Return assets matching a category."""
    cat = category.lower()
    return [a for a in build_full_registry() if a.category == cat]


def get_categories() -> List[str]:
    """List of all available categories."""
    return ["equity", "sector", "commodity", "forex", "bond", "index"]


def to_dict_records() -> list:
    """Flat dict list useful for DataFrame construction."""
    return [
        {
            "ticker": a.ticker,
            "name": a.name,
            "category": a.category,
            "subcategory": a.subcategory,
            "fetch_symbol": a.fetch_symbol(),
        }
        for a in build_full_registry()
    ]


def fetch_full_us_equities(provider: str = "yfinance") -> List[str]:
    """Return ticker list for the full US equity universe.

    For `polygon` or `alpaca`, this uses the API to enumerate all active
    common stocks. For `yfinance`, falls back to a Wikipedia-scraped S&P 500.
    """
    if provider == "polygon":
        from data.market_fetcher import _polygon_full_us_tickers
        return _polygon_full_us_tickers()
    if provider == "alpaca":
        from data.market_fetcher import _alpaca_full_us_tickers
        return _alpaca_full_us_tickers()
    # yfinance fallback — S&P 500
    from data.market_fetcher import _sp500_tickers
    return _sp500_tickers()
