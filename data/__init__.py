"""Live data fetchers: macro (FRED), market (yfinance/Polygon/Alpaca), news (NewsAPI)."""

from .macro_fetcher import fetch_macro_snapshot, fetch_macro_history, FRED_SERIES
from .market_fetcher import fetch_market_snapshot, fetch_ticker_universe
from .news_fetcher import fetch_market_news
