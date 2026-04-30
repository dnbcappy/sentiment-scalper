"""Tests for index.py — Bull/Bear Index composite score."""

from __future__ import annotations

from datetime import datetime, timezone

from index import (
    BEAR_THRESHOLD,
    BULL_THRESHOLD,
    color_for,
    compute_bull_bear_index,
    label_for,
)
from tests.conftest import seed_mentions


def test_index_empty_db(db_path):
    """No mentions in window => score is None, n is 0."""
    out = compute_bull_bear_index(hours=24)
    assert out["score"] is None
    assert out["n"] == 0
    assert out["bullish"] == 0
    assert out["bearish"] == 0


def test_index_strong_bullish_window(db_path):
    """All mentions strongly positive => score firmly in greed territory."""
    now = int(datetime.now(timezone.utc).timestamp())
    seed_mentions(
        db_path,
        [{"ticker": "BTC", "created_utc": now - 60, "compound": 0.8} for _ in range(20)],
    )
    out = compute_bull_bear_index(hours=24)
    assert out["score"] is not None
    assert out["score"] > 70  # firmly greed/extreme greed
    assert out["bullish"] == 20
    assert out["bearish"] == 0


def test_index_strong_bearish_window(db_path):
    """All mentions strongly negative => score firmly in fear territory."""
    now = int(datetime.now(timezone.utc).timestamp())
    seed_mentions(
        db_path,
        [{"ticker": "BTC", "created_utc": now - 60, "compound": -0.8} for _ in range(20)],
    )
    out = compute_bull_bear_index(hours=24)
    assert out["score"] is not None
    assert out["score"] < 30  # firmly fear/extreme fear
    assert out["bearish"] == 20
    assert out["bullish"] == 0


def test_index_neutral_balance(db_path):
    """Equal bull/bear, near-zero avg => score near 50."""
    now = int(datetime.now(timezone.utc).timestamp())
    rows = []
    for _ in range(10):
        rows.append({"ticker": "BTC", "created_utc": now - 60, "compound": 0.3})
    for _ in range(10):
        rows.append({"ticker": "BTC", "created_utc": now - 60, "compound": -0.3})
    seed_mentions(db_path, rows)
    out = compute_bull_bear_index(hours=24)
    assert out["score"] is not None
    assert 40 <= out["score"] <= 60
    assert out["bullish"] == 10
    assert out["bearish"] == 10


def test_index_window_filter(db_path):
    """Mentions outside the window are not counted."""
    now = int(datetime.now(timezone.utc).timestamp())
    rows = [
        # In window: last 24h
        {"ticker": "BTC", "created_utc": now - 60, "compound": 0.8},
        # Outside window: 30 hours ago
        {"ticker": "BTC", "created_utc": now - 30 * 3600, "compound": -0.8},
    ]
    seed_mentions(db_path, rows)
    out = compute_bull_bear_index(hours=24, ago_hours=0)
    assert out["n"] == 1
    assert out["bullish"] == 1
    assert out["bearish"] == 0


def test_index_ago_hours_shifts_window(db_path):
    """ago_hours=24 should move the window to 24-48h ago."""
    now = int(datetime.now(timezone.utc).timestamp())
    rows = [
        # 36h ago — falls inside the "yesterday" 24-48h window
        {"ticker": "BTC", "created_utc": now - 36 * 3600, "compound": 0.8},
        # 12h ago — falls inside "today" but not "yesterday"
        {"ticker": "BTC", "created_utc": now - 12 * 3600, "compound": -0.8},
    ]
    seed_mentions(db_path, rows)
    yesterday = compute_bull_bear_index(hours=24, ago_hours=24)
    today = compute_bull_bear_index(hours=24, ago_hours=0)
    assert yesterday["n"] == 1
    assert yesterday["bullish"] == 1
    assert today["n"] == 1
    assert today["bearish"] == 1


def test_index_model_filter(db_path):
    """Only mentions from the given model contribute."""
    now = int(datetime.now(timezone.utc).timestamp())
    rows = [
        {"ticker": "BTC", "created_utc": now - 60, "compound": 0.8, "model": "vader"},
        {"ticker": "BTC", "created_utc": now - 60, "compound": -0.8, "model": "finbert"},
    ]
    seed_mentions(db_path, rows)
    vader = compute_bull_bear_index(hours=24, model="vader")
    finbert = compute_bull_bear_index(hours=24, model="finbert")
    assert vader["bullish"] == 1 and vader["bearish"] == 0
    assert finbert["bullish"] == 0 and finbert["bearish"] == 1


def test_label_for_bands():
    assert label_for(None) == "—"
    assert label_for(10) == "Extreme Fear"
    assert label_for(35) == "Fear"
    assert label_for(50) == "Neutral"
    assert label_for(65) == "Greed"
    assert label_for(90) == "Extreme Greed"


def test_color_for_bands():
    """Sanity check that color_for returns a hex color and bands transition."""
    assert color_for(None).startswith("#")
    assert color_for(10) != color_for(50)
    assert color_for(50) != color_for(90)


def test_thresholds_are_sane():
    """Sanity check the constants."""
    assert BULL_THRESHOLD > 0
    assert BEAR_THRESHOLD < 0
    assert BULL_THRESHOLD == -BEAR_THRESHOLD  # symmetric, by convention
