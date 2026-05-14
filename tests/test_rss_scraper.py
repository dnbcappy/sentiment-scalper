"""Tests for rss_scraper.py — feed parsing and dict shape.

The actual RSS feeds aren't fetched in unit tests (network dependent). We
mock feedparser.parse and assert:
  - Items come out in the expected dict shape
  - Old entries (outside LOOKBACK_HOURS) are filtered out
  - HTML in summaries is stripped
  - A failing feed doesn't kill subsequent feeds
"""

from __future__ import annotations

import calendar
import sys
import time
from datetime import datetime, timedelta, timezone
from types import SimpleNamespace
from unittest.mock import MagicMock


def _struct_for(dt_utc: datetime) -> time.struct_time:
    """feedparser gives published_parsed/updated_parsed as a UTC struct_time."""
    return dt_utc.timetuple()


# Real feedparser entries are FeedParserDict objects supporting BOTH .get()
# and attribute access. The code uses .get(), so test fixtures are plain
# dicts (SimpleNamespace lacks .get()).
def _entry(link, title, summary, dt_utc=None):
    e = {"link": link, "title": title, "summary": summary}
    if dt_utc is not None:
        e["published_parsed"] = _struct_for(dt_utc)
    return e


def test_fetch_rss_returns_items_in_expected_shape(monkeypatch):
    now_utc = datetime.now(timezone.utc)
    one_hour_ago = now_utc - timedelta(hours=1)

    entry = _entry(
        "https://example.com/btc-up",
        "BTC pumps to new highs",
        "<p>Bitcoin <b>broke out</b> overnight.</p>",
        one_hour_ago,
    )

    fake_parsed = SimpleNamespace(entries=[entry])
    fake_module = MagicMock()
    fake_module.parse.return_value = fake_parsed
    monkeypatch.setitem(sys.modules, "feedparser", fake_module)

    monkeypatch.setattr("rss_scraper.RSS_FEEDS", {"TestFeed": "https://example.com/rss"})

    from rss_scraper import fetch_rss

    items = fetch_rss()

    assert len(items) == 1
    item = items[0]
    assert item["id"].startswith("rss_")
    assert item["source"] == "rss"
    assert item["subreddit"] == "TestFeed"
    assert "BTC pumps to new highs" in item["text"]
    assert "<p>" not in item["text"]
    assert "<b>" not in item["text"]
    assert "Bitcoin broke out overnight." in item["text"]
    assert item["score"] == 0
    assert item["created_utc"] == calendar.timegm(one_hour_ago.timetuple())


def test_fetch_rss_filters_old_entries(monkeypatch):
    now_utc = datetime.now(timezone.utc)
    fresh = now_utc - timedelta(hours=1)
    too_old = now_utc - timedelta(hours=200)  # > LOOKBACK_HOURS

    entries = [
        _entry("https://example.com/fresh", "fresh news", "recent", fresh),
        _entry("https://example.com/old", "old news", "old", too_old),
    ]
    fake_parsed = SimpleNamespace(entries=entries)
    fake_module = MagicMock()
    fake_module.parse.return_value = fake_parsed
    monkeypatch.setitem(sys.modules, "feedparser", fake_module)
    monkeypatch.setattr("rss_scraper.RSS_FEEDS", {"TestFeed": "https://example.com/rss"})

    from rss_scraper import fetch_rss

    items = fetch_rss()
    assert len(items) == 1
    assert "fresh" in items[0]["text"]


def test_fetch_rss_skips_entries_without_dates(monkeypatch):
    """Entries missing both published_parsed and updated_parsed are skipped."""
    # _entry without dt_utc omits the published_parsed key entirely
    entry = _entry("https://example.com/dateless", "undated", "no date")
    fake_parsed = SimpleNamespace(entries=[entry])
    fake_module = MagicMock()
    fake_module.parse.return_value = fake_parsed
    monkeypatch.setitem(sys.modules, "feedparser", fake_module)
    monkeypatch.setattr("rss_scraper.RSS_FEEDS", {"TestFeed": "https://example.com/rss"})

    from rss_scraper import fetch_rss

    assert fetch_rss() == []


def test_fetch_rss_dedupes_same_link_within_run(monkeypatch):
    now_utc = datetime.now(timezone.utc)
    one_hour_ago = now_utc - timedelta(hours=1)

    # Same link returned twice within a single feed AND also from a second feed.
    # All should collapse to one item.
    duplicate = _entry("https://example.com/same", "repeated", "x", one_hour_ago)

    fake_parsed = SimpleNamespace(entries=[duplicate, duplicate])
    fake_module = MagicMock()
    fake_module.parse.return_value = fake_parsed
    monkeypatch.setitem(sys.modules, "feedparser", fake_module)
    monkeypatch.setattr("rss_scraper.RSS_FEEDS", {"A": "x", "B": "y"})

    from rss_scraper import fetch_rss

    items = fetch_rss()
    assert len(items) == 1


def test_fetch_rss_handles_feed_error_gracefully(monkeypatch):
    """A failing feed shouldn't kill subsequent feeds."""
    now_utc = datetime.now(timezone.utc)
    good_entry = _entry(
        "https://example.com/good", "works", "ok", now_utc - timedelta(hours=1)
    )

    fake_module = MagicMock()

    def parse_side_effect(url):
        if "broken" in url:
            raise RuntimeError("simulated network failure")
        return SimpleNamespace(entries=[good_entry])

    fake_module.parse.side_effect = parse_side_effect
    monkeypatch.setitem(sys.modules, "feedparser", fake_module)
    monkeypatch.setattr(
        "rss_scraper.RSS_FEEDS",
        {"broken": "https://broken.com/rss", "working": "https://working.com/rss"},
    )

    from rss_scraper import fetch_rss

    items = fetch_rss()
    assert len(items) == 1
    assert items[0]["subreddit"] == "working"
