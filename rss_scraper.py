"""
Sentiment Scalper - RSS ingestion
----------------------------------
Pulls recent articles from a curated set of finance RSS feeds via feedparser.
Returns dicts in the same shape as fetch_newsapi() so the rest of the ingest
pipeline doesn't care which source the data came from.

Why RSS instead of, say, Reddit:
  - RSS is explicitly designed for programmatic consumption (no ToS friction)
  - No API key, no rate limit, no approval process
  - Publishers actively publish RSS so you can pull from them
  - Same financial publishers we already trust for NewsAPI

Volume vs. NewsAPI:
  Each feed serves 20-50 of its most recent articles per fetch. With 6 feeds
  configured, expect ~150-300 articles per cron run — order of magnitude more
  than NewsAPI's ~9-per-query budget. Net new articles after dedup against
  NewsAPI's overlapping coverage of the same publishers: roughly 50-150 per
  run. Together they comfortably outpace NewsAPI's 100/day rate limit.

Within-run dedup: returns items with id format `rss_<md5(url)[:16]>`. The
caller (sentiment_scalper.ingest) deduplicates by URL hash across both
NewsAPI and RSS within the same run, so the same article doesn't get scored
and inserted twice in one pass.
"""

from __future__ import annotations

import calendar
import hashlib
import logging
import re
from datetime import datetime, timedelta, timezone

logger = logging.getLogger(__name__)

# Curated mix of stock + crypto financial publishers. Mostly overlaps with
# our NewsAPI domain whitelist (the same publishers we already trust), with
# Investing.com added for extra non-overlapping coverage.
RSS_FEEDS: dict[str, str] = {
    "CoinDesk": "https://www.coindesk.com/arc/outboundfeeds/rss/",
    "Cointelegraph": "https://cointelegraph.com/rss",
    "Decrypt": "https://decrypt.co/feed",
    "CNBC": "https://www.cnbc.com/id/100003114/device/rss/rss.html",
    "MarketWatch": "https://feeds.marketwatch.com/marketwatch/topstories/",
    "Investing.com": "https://www.investing.com/rss/news.rss",
}

LOOKBACK_HOURS = 168

_HTML_TAG = re.compile(r"<[^>]+>")


def fetch_rss() -> list[dict]:
    """Pull recent articles from configured RSS feeds.

    Returns dicts shaped like fetch_newsapi() output:
        {id, source, subreddit, text, score, created_utc}

    `subreddit` is repurposed as the feed/publisher name (matches our
    NewsAPI convention where it stores the publisher).
    """
    import feedparser

    cutoff_ts = int((datetime.now(timezone.utc) - timedelta(hours=LOOKBACK_HOURS)).timestamp())
    items: list[dict] = []
    seen_links: set[str] = set()

    for feed_name, feed_url in RSS_FEEDS.items():
        accepted = 0
        try:
            parsed = feedparser.parse(feed_url)
            for entry in parsed.entries:
                link = (entry.get("link") or "").strip()
                if not link or link in seen_links:
                    continue

                # feedparser exposes the publish/update date as a struct_time
                # in UTC. calendar.timegm converts UTC struct_time -> epoch
                # (time.mktime would assume local time, which is wrong).
                published_struct = entry.get("published_parsed") or entry.get("updated_parsed")
                if not published_struct:
                    continue
                created = calendar.timegm(published_struct)

                if created < cutoff_ts:
                    continue

                title = (entry.get("title") or "").strip()
                summary_raw = entry.get("summary") or entry.get("description") or ""
                # Feeds vary: some return clean text, some HTML. Strip tags
                # so the sentiment engine doesn't waste cycles on markup.
                summary = _HTML_TAG.sub("", summary_raw).strip()

                seen_links.add(link)
                items.append(
                    {
                        "id": f"rss_{hashlib.md5(link.encode()).hexdigest()[:16]}",
                        "source": "rss",
                        "subreddit": feed_name,
                        "text": f"{title}\n\n{summary}".strip(),
                        "score": 0,
                        "created_utc": created,
                    }
                )
                accepted += 1
            logger.info("Fetched %d entries from %s", accepted, feed_name)
        except Exception:
            # Don't pass the exception object — feedparser errors can include
            # the feed URL with embedded credentials on some private feeds.
            logger.error("Failed to fetch from %s", feed_name)

    logger.info("RSS fetch totals: %d entries across %d feeds", len(items), len(RSS_FEEDS))
    return items
