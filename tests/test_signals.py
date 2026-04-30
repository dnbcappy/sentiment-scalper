"""Tests for signals.py — z-score signal engine."""

from __future__ import annotations

from datetime import datetime, timezone

from signals import (
    BASELINE_HOURS,
    MIN_HIST_BUCKETS,
    VOLUME_Z_MIN,
    WINDOW_HOURS,
    compute_historical_signals,
    compute_signals,
    list_engines,
)
from tests.conftest import baseline_rows, seed_mentions

WINDOW_SECS = WINDOW_HOURS * 3600


# ---------- list_engines ----------


def test_list_engines_empty(db_path):
    assert list_engines() == []


def test_list_engines_returns_distinct_sorted(db_path):
    now = int(datetime.now(timezone.utc).timestamp())
    seed_mentions(
        db_path,
        [
            {"ticker": "BTC", "created_utc": now, "compound": 0.0, "model": "vader"},
            {"ticker": "BTC", "created_utc": now, "compound": 0.0, "model": "finbert"},
            {"ticker": "ETH", "created_utc": now, "compound": 0.0, "model": "vader"},
        ],
    )
    assert list_engines() == ["finbert", "vader"]


# ---------- compute_signals ----------


def test_compute_signals_empty_db(db_path):
    df = compute_signals()
    assert df.empty
    for col in ["ticker", "direction", "signal", "volume_z", "sentiment_z"]:
        assert col in df.columns


def test_compute_signals_skips_insufficient_history(db_path):
    """Tickers with fewer than MIN_HIST_BUCKETS completed buckets are skipped."""
    now_ts = int(datetime.now(timezone.utc).timestamp())
    current_bucket = now_ts // WINDOW_SECS
    rows = []
    for offset in (1, 2):
        bucket_ts = (current_bucket - offset) * WINDOW_SECS + 100
        rows.append({"ticker": "BTC", "created_utc": bucket_ts, "compound": 0.0})
    seed_mentions(db_path, rows)
    df = compute_signals()
    assert df.empty


def test_compute_signals_volume_spike_triggers_bull_signal(db_path):
    """Anomalous volume + positive sentiment in the active bucket fires BULL."""
    now_ts = int(datetime.now(timezone.utc).timestamp())
    current_bucket = now_ts // WINDOW_SECS

    rows = baseline_rows("BTC", current_bucket, WINDOW_SECS)
    rows.extend({"ticker": "BTC", "created_utc": now_ts, "compound": 0.7} for _ in range(15))
    seed_mentions(db_path, rows)

    df = compute_signals()
    assert len(df) == 1
    btc = df.iloc[0]
    assert btc["ticker"] == "BTC"
    assert btc["volume_z"] > VOLUME_Z_MIN
    assert btc["sentiment_z"] > 0
    assert btc["signal"] > 0
    assert btc["direction"] == "BULL"


def test_compute_signals_volume_spike_triggers_bear_signal(db_path):
    """Anomalous volume + negative sentiment fires BEAR."""
    now_ts = int(datetime.now(timezone.utc).timestamp())
    current_bucket = now_ts // WINDOW_SECS

    rows = baseline_rows("BTC", current_bucket, WINDOW_SECS)
    rows.extend({"ticker": "BTC", "created_utc": now_ts, "compound": -0.7} for _ in range(15))
    seed_mentions(db_path, rows)

    df = compute_signals()
    assert len(df) == 1
    assert df.iloc[0]["signal"] < 0
    assert df.iloc[0]["direction"] == "BEAR"


def test_compute_signals_quiet_active_bucket_no_signal(db_path):
    """Active bucket with no mentions => sentiment_z=0 => signal=0 regardless
    of how anomalous the volume is. (Pro-rating means raw count is no longer
    a stable proxy for 'volume gate' behavior; sentiment direction is.)"""
    now_ts = int(datetime.now(timezone.utc).timestamp())
    current_bucket = now_ts // WINDOW_SECS

    rows = baseline_rows("BTC", current_bucket, WINDOW_SECS)
    # No active-bucket mentions at all
    seed_mentions(db_path, rows)
    df = compute_signals()
    if not df.empty:
        assert df.iloc[0]["signal"] == 0.0


def test_signal_at_bucket_pro_rates_count_by_elapsed(db_path):
    """Direct unit test: _signal_at_bucket(elapsed_fraction=0.5) should pro-rate
    the active bucket's count by 2x relative to elapsed_fraction=1.0."""
    from datetime import datetime, timezone

    from signals import _signal_at_bucket

    now_ts = int(datetime.now(timezone.utc).timestamp())
    current_bucket = now_ts // WINDOW_SECS

    # Build an in-memory DataFrame matching what _load returns
    import pandas as pd

    rows = baseline_rows("BTC", current_bucket, WINDOW_SECS)
    rows.extend({"ticker": "BTC", "created_utc": now_ts, "compound": 0.7} for _ in range(3))
    df = pd.DataFrame(rows)
    df["bucket"] = df["created_utc"] // WINDOW_SECS

    s_full = _signal_at_bucket(df, current_bucket, elapsed_fraction=1.0)
    s_half = _signal_at_bucket(df, current_bucket, elapsed_fraction=0.5)

    assert s_full is not None and s_half is not None
    # current_count (raw) is the same — pro-rating doesn't change the display value
    assert s_half["current_count"] == s_full["current_count"] == 3
    # but volume_z grows when elapsed_fraction shrinks (same count, more weight)
    assert s_half["volume_z"] > s_full["volume_z"]


def test_compute_signals_model_filter(db_path):
    """Only mentions matching the given model are considered."""
    now_ts = int(datetime.now(timezone.utc).timestamp())
    current_bucket = now_ts // WINDOW_SECS

    rows = baseline_rows("BTC", current_bucket, WINDOW_SECS, model="vader")
    rows.extend(
        {"ticker": "BTC", "created_utc": now_ts, "compound": 0.7, "model": "vader"}
        for _ in range(15)
    )
    rows.append({"ticker": "BTC", "created_utc": now_ts, "compound": 0.0, "model": "finbert"})
    seed_mentions(db_path, rows)

    vader = compute_signals(model="vader")
    finbert = compute_signals(model="finbert")
    assert len(vader) == 1
    assert vader.iloc[0]["direction"] == "BULL"
    assert finbert.empty


# ---------- compute_historical_signals ----------


def test_compute_historical_signals_empty(db_path):
    df = compute_historical_signals(threshold=1.5)
    assert df.empty
    assert list(df.columns) == ["ticker", "signal_ts", "signal", "direction"]


def test_compute_historical_signals_excludes_active_bucket(db_path):
    """The active (current) bucket is never emitted as a historical signal."""
    now_ts = int(datetime.now(timezone.utc).timestamp())
    current_bucket = now_ts // WINDOW_SECS

    rows = baseline_rows("BTC", current_bucket, WINDOW_SECS)
    rows.extend({"ticker": "BTC", "created_utc": now_ts, "compound": 0.7} for _ in range(20))
    seed_mentions(db_path, rows)

    df = compute_historical_signals(threshold=0.0)
    active_end_ts = (current_bucket + 1) * WINDOW_SECS
    if not df.empty:
        assert (df["signal_ts"] < active_end_ts).all()


def test_constants_are_consistent():
    """Quick sanity check that the constants align with the docstring."""
    assert BASELINE_HOURS % WINDOW_HOURS == 0, "BASELINE_HOURS must be a multiple of WINDOW_HOURS"
    assert MIN_HIST_BUCKETS >= 2, "Need >=2 buckets for a meaningful std"
