"""Tests for reddit_scraper.py — credential handling and dict shape.

The real PRAW client + Reddit API are not exercised in unit tests (network +
auth dependent). We assert:
  - Missing credentials => empty list (graceful degradation)
  - When PRAW is mocked, the function emits items in the expected shape
"""

from __future__ import annotations

import sys
from datetime import datetime, timezone
from types import SimpleNamespace
from unittest.mock import MagicMock


def test_fetch_reddit_returns_empty_without_credentials(monkeypatch):
    monkeypatch.delenv("REDDIT_CLIENT_ID", raising=False)
    monkeypatch.delenv("REDDIT_CLIENT_SECRET", raising=False)

    from reddit_scraper import fetch_reddit

    assert fetch_reddit() == []


def test_fetch_reddit_returns_empty_when_client_id_blank(monkeypatch):
    monkeypatch.setenv("REDDIT_CLIENT_ID", "   ")  # whitespace-only counts as missing
    monkeypatch.setenv("REDDIT_CLIENT_SECRET", "abc")

    from reddit_scraper import fetch_reddit

    assert fetch_reddit() == []


def _make_fake_post(post_id: str, title: str, body: str, created_utc: int, score: int = 5):
    return SimpleNamespace(
        id=post_id,
        title=title,
        selftext=body,
        created_utc=float(created_utc),
        score=score,
    )


def test_fetch_reddit_shapes_items_correctly(monkeypatch):
    monkeypatch.setenv("REDDIT_CLIENT_ID", "fake-id")
    monkeypatch.setenv("REDDIT_CLIENT_SECRET", "fake-secret")

    now_ts = int(datetime.now(timezone.utc).timestamp())
    fake_post = _make_fake_post("abc123", "Tesla earnings beat", "TSLA stock rallies", now_ts)

    fake_subreddit = MagicMock()
    fake_subreddit.new.return_value = iter([fake_post])

    fake_reddit_instance = MagicMock()
    fake_reddit_instance.subreddit.return_value = fake_subreddit

    fake_praw_module = MagicMock()
    fake_praw_module.Reddit.return_value = fake_reddit_instance
    monkeypatch.setitem(sys.modules, "praw", fake_praw_module)

    # Force the function to only iterate one subreddit by patching the list
    monkeypatch.setattr("reddit_scraper.REDDIT_SUBREDDITS", ["wallstreetbets"])

    from reddit_scraper import fetch_reddit

    items = fetch_reddit()

    assert len(items) == 1
    item = items[0]
    assert item["id"] == "reddit_abc123"
    assert item["source"] == "reddit"
    assert item["subreddit"] == "wallstreetbets"
    assert "Tesla earnings beat" in item["text"]
    assert "TSLA stock rallies" in item["text"]
    assert item["created_utc"] == now_ts
    assert item["score"] == 5


def test_fetch_reddit_skips_posts_older_than_lookback(monkeypatch):
    monkeypatch.setenv("REDDIT_CLIENT_ID", "fake-id")
    monkeypatch.setenv("REDDIT_CLIENT_SECRET", "fake-secret")

    now_ts = int(datetime.now(timezone.utc).timestamp())
    # 200h ago — older than the 168h LOOKBACK_HOURS
    too_old = now_ts - 200 * 3600
    fresh = now_ts - 24 * 3600  # 1 day ago, in window

    posts = [
        _make_fake_post("fresh01", "fresh title", "AAPL up", fresh),
        _make_fake_post("oldold02", "old title", "BTC down", too_old),
    ]

    fake_subreddit = MagicMock()
    fake_subreddit.new.return_value = iter(posts)

    fake_reddit_instance = MagicMock()
    fake_reddit_instance.subreddit.return_value = fake_subreddit

    fake_praw_module = MagicMock()
    fake_praw_module.Reddit.return_value = fake_reddit_instance
    monkeypatch.setitem(sys.modules, "praw", fake_praw_module)
    monkeypatch.setattr("reddit_scraper.REDDIT_SUBREDDITS", ["test"])

    from reddit_scraper import fetch_reddit

    items = fetch_reddit()

    # Only the fresh post should be accepted
    assert len(items) == 1
    assert items[0]["id"] == "reddit_fresh01"


def test_fetch_reddit_handles_subreddit_error_gracefully(monkeypatch):
    """If one subreddit errors, other subreddits should still be tried."""
    monkeypatch.setenv("REDDIT_CLIENT_ID", "fake-id")
    monkeypatch.setenv("REDDIT_CLIENT_SECRET", "fake-secret")

    now_ts = int(datetime.now(timezone.utc).timestamp())
    good_post = _make_fake_post("good01", "AAPL update", "Apple stock news", now_ts)

    def subreddit_factory(name):
        sub = MagicMock()
        if name == "broken":
            sub.new.side_effect = RuntimeError("simulated PRAW failure")
        else:
            sub.new.return_value = iter([good_post])
        return sub

    fake_reddit_instance = MagicMock()
    fake_reddit_instance.subreddit.side_effect = subreddit_factory

    fake_praw_module = MagicMock()
    fake_praw_module.Reddit.return_value = fake_reddit_instance
    monkeypatch.setitem(sys.modules, "praw", fake_praw_module)
    monkeypatch.setattr("reddit_scraper.REDDIT_SUBREDDITS", ["broken", "working"])

    from reddit_scraper import fetch_reddit

    items = fetch_reddit()

    # Broken subreddit fails silently, working subreddit still produces an item
    assert len(items) == 1
    assert items[0]["subreddit"] == "working"
