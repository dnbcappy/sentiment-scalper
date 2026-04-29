"""Tests for backtest.py — hit rate computation against historical signals."""

from __future__ import annotations

from datetime import datetime, timezone

from backtest import compute_hit_rates
from signals import WINDOW_HOURS
from tests.conftest import baseline_rows, seed_mentions, seed_prices

WINDOW_SECS = WINDOW_HOURS * 3600


def test_compute_hit_rates_empty_db(db_path):
    """No data anywhere => empty (but well-typed) DataFrame."""
    df = compute_hit_rates(db_path, threshold=1.5)
    assert df.empty
    expected_cols = {"ticker", "n_signals", "hit_rate_1d", "hit_rate_3d", "hit_rate_7d"}
    assert expected_cols.issubset(set(df.columns))


def test_compute_hit_rates_no_prices(db_path):
    """Signals exist but no price data => empty (cannot evaluate hits)."""
    now_ts = int(datetime.now(timezone.utc).timestamp())
    current_bucket = now_ts // WINDOW_SECS

    rows = baseline_rows("BTC", current_bucket, WINDOW_SECS)
    # Volume spike in last completed bucket — generates a historical signal
    spike_bucket_ts = (current_bucket - 1) * WINDOW_SECS + 100
    rows.extend(
        {"ticker": "BTC", "created_utc": spike_bucket_ts, "compound": 0.7} for _ in range(15)
    )
    seed_mentions(db_path, rows)
    df = compute_hit_rates(db_path, threshold=1.5)
    assert df.empty


def test_hit_rate_bull_signal_with_price_rise(db_path):
    """A BULL historical signal followed by a price rise should count as a hit."""
    now_ts = int(datetime.now(timezone.utc).timestamp())
    current_bucket = now_ts // WINDOW_SECS
    # Place the signal far enough back that all horizons elapse in seeded prices
    sig_bucket = current_bucket - 30

    rows = baseline_rows("BTC", sig_bucket, WINDOW_SECS)
    rows.extend(
        {"ticker": "BTC", "created_utc": sig_bucket * WINDOW_SECS + 100, "compound": 0.7}
        for _ in range(15)
    )
    seed_mentions(db_path, rows)

    signal_ts = (sig_bucket + 1) * WINDOW_SECS
    seed_prices(
        db_path,
        [
            {"ticker": "BTC", "ts": signal_ts - 100, "close": 100.0},
            {"ticker": "BTC", "ts": signal_ts + 86400, "close": 110.0},
            {"ticker": "BTC", "ts": signal_ts + 3 * 86400, "close": 120.0},
            {"ticker": "BTC", "ts": signal_ts + 7 * 86400, "close": 130.0},
            # Latest bar must be after the 7d horizon to count it
            {"ticker": "BTC", "ts": signal_ts + 8 * 86400, "close": 130.0},
        ],
    )
    df = compute_hit_rates(db_path, threshold=1.5)
    assert not df.empty
    btc = df[df["ticker"] == "BTC"].iloc[0]
    assert btc["n_signals"] >= 1
    assert btc["hit_rate_1d"] == 1.0
    assert btc["hit_rate_3d"] == 1.0
    assert btc["hit_rate_7d"] == 1.0
    assert btc["avg_return_1d"] > 0
    assert btc["avg_return_7d"] > 0


def test_hit_rate_unelapsed_horizon_is_excluded(db_path):
    """If exit_ts > latest price bar, that horizon is None (still-open position)."""
    now_ts = int(datetime.now(timezone.utc).timestamp())
    current_bucket = now_ts // WINDOW_SECS
    sig_bucket = current_bucket - 1

    rows = baseline_rows("BTC", sig_bucket, WINDOW_SECS)
    rows.extend(
        {"ticker": "BTC", "created_utc": sig_bucket * WINDOW_SECS + 100, "compound": 0.7}
        for _ in range(15)
    )
    seed_mentions(db_path, rows)

    signal_ts = (sig_bucket + 1) * WINDOW_SECS
    # Only seed a single price bar AT the signal time — no future data
    seed_prices(
        db_path,
        [{"ticker": "BTC", "ts": signal_ts - 100, "close": 100.0}],
    )
    df = compute_hit_rates(db_path, threshold=1.5)
    assert not df.empty
    btc = df[df["ticker"] == "BTC"].iloc[0]
    assert btc["hit_rate_1d"] is None
    assert btc["hit_rate_3d"] is None
    assert btc["hit_rate_7d"] is None


def test_hit_rate_includes_all_aggregate_row(db_path):
    """The 'ALL' row appears alongside per-ticker rows."""
    now_ts = int(datetime.now(timezone.utc).timestamp())
    current_bucket = now_ts // WINDOW_SECS
    sig_bucket = current_bucket - 30

    rows = baseline_rows("BTC", sig_bucket, WINDOW_SECS)
    rows.extend(
        {"ticker": "BTC", "created_utc": sig_bucket * WINDOW_SECS + 100, "compound": 0.7}
        for _ in range(15)
    )
    seed_mentions(db_path, rows)

    signal_ts = (sig_bucket + 1) * WINDOW_SECS
    seed_prices(
        db_path,
        [
            {"ticker": "BTC", "ts": signal_ts - 100, "close": 100.0},
            {"ticker": "BTC", "ts": signal_ts + 86400, "close": 110.0},
            {"ticker": "BTC", "ts": signal_ts + 8 * 86400, "close": 120.0},
        ],
    )
    df = compute_hit_rates(db_path, threshold=1.5)
    tickers = set(df["ticker"])
    assert "ALL" in tickers
    assert "BTC" in tickers
