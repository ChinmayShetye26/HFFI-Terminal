"""
News fetcher.

Pulls recent market and macro news headlines for the terminal's news panel.
Uses NewsAPI free tier (https://newsapi.org/) by default. Falls back to a
direct RSS scrape of Reuters / WSJ market headlines if no key is set.

NETWORK NOTE: Live HTTP. Will not work in sandboxes without network.
"""

from __future__ import annotations
from datetime import datetime, timedelta
from pathlib import Path
from typing import List, Dict, Optional
import os
import logging
import pandas as pd

logger = logging.getLogger(__name__)


DEFAULT_QUERIES = [
    "Federal Reserve interest rate",
    "inflation CPI",
    "stock market",
    "unemployment jobs report",
    "mortgage rates housing",
]


def fetch_market_news(
    queries: Optional[List[str]] = None,
    page_size: int = 10,
    use_cache: bool = True,
    cache_minutes: int = 30,
) -> pd.DataFrame:
    """Fetch recent market news headlines.

    Returns a DataFrame with columns: title, source, url, published, query,
    description.
    """
    queries = queries or DEFAULT_QUERIES
    cache_path = Path(".cache/news.parquet")
    if use_cache and cache_path.exists():
        mtime = datetime.fromtimestamp(cache_path.stat().st_mtime)
        if datetime.now() - mtime < timedelta(minutes=cache_minutes):
            logger.info("Using cached news")
            return pd.read_parquet(cache_path)

    api_key = os.getenv("NEWSAPI_KEY")
    if api_key:
        df = _fetch_via_newsapi(queries, page_size, api_key)
    else:
        logger.info(
            "NEWSAPI_KEY not set — using RSS fallback (limited coverage). "
            "Get a free key at https://newsapi.org/."
        )
        df = _fetch_via_rss()

    cache_path.parent.mkdir(parents=True, exist_ok=True)
    df.to_parquet(cache_path)
    return df


def _fetch_via_newsapi(queries: List[str], page_size: int, api_key: str) -> pd.DataFrame:
    """NewsAPI everything endpoint, one query at a time."""
    import requests
    rows = []
    for q in queries:
        try:
            r = requests.get(
                "https://newsapi.org/v2/everything",
                params={
                    "q": q,
                    "language": "en",
                    "sortBy": "publishedAt",
                    "pageSize": page_size,
                    "apiKey": api_key,
                },
                timeout=10,
            )
            if r.status_code != 200:
                logger.warning("NewsAPI %s for %r", r.status_code, q)
                continue
            for art in r.json().get("articles", []):
                rows.append({
                    "title":       art.get("title", ""),
                    "source":      (art.get("source") or {}).get("name", ""),
                    "url":         art.get("url", ""),
                    "published":   art.get("publishedAt", ""),
                    "description": art.get("description", "") or "",
                    "query":       q,
                })
        except Exception as e:
            logger.warning("NewsAPI fetch failed for %r: %s", q, e)
    return pd.DataFrame(rows)


def _fetch_via_rss() -> pd.DataFrame:
    """Fallback: scrape a few public RSS feeds. Limited but no key needed."""
    import requests
    import xml.etree.ElementTree as ET

    feeds = {
        "Reuters Markets": "https://www.reutersagency.com/feed/?best-topics=business-finance&post_type=best",
        "MarketWatch":     "https://feeds.marketwatch.com/marketwatch/topstories/",
        "CNBC Markets":    "https://www.cnbc.com/id/10000664/device/rss/rss.html",
    }
    rows = []
    for source, url in feeds.items():
        try:
            r = requests.get(url, timeout=10, headers={"User-Agent": "Mozilla/5.0"})
            if r.status_code != 200:
                continue
            root = ET.fromstring(r.content)
            for item in root.iter("item"):
                rows.append({
                    "title":       (item.findtext("title") or "").strip(),
                    "source":      source,
                    "url":         (item.findtext("link") or "").strip(),
                    "published":   (item.findtext("pubDate") or "").strip(),
                    "description": (item.findtext("description") or "").strip()[:280],
                    "query":       "rss",
                })
        except Exception as e:
            logger.warning("RSS fetch failed for %s: %s", source, e)
    return pd.DataFrame(rows)
