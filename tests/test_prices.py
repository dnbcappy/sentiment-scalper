"""Tests for prices.py — SQL helpers and safe-float coercion.

We do not exercise yfinance directly; that's a network-dependent integration
concern handled outside the unit suite.
"""

from __future__ import annotations

from prices import _f, get_prices
from tests.conftest import seed_prices


def test_get_prices_no_table_returns_empty(db_path):
    """If the prices table has never been created, get_prices returns empty."""
    df = get_prices(ticker="BTC")
    assert df.empty


def test_get_prices_filters_by_ticker(db_path):
    seed_prices(
        db_path,
        [
            {"ticker": "BTC", "ts": 1700000000, "close": 50000.0},
            {"ticker": "ETH", "ts": 1700000000, "close": 3000.0},
            {"ticker": "BTC", "ts": 1700086400, "close": 51000.0},
        ],
    )
    btc = get_prices(ticker="BTC")
    assert len(btc) == 2
    assert set(btc["ticker"]) == {"BTC"}


def test_get_prices_filters_by_since_ts(db_path):
    seed_prices(
        db_path,
        [
            {"ticker": "BTC", "ts": 1700000000, "close": 50000.0},
            {"ticker": "BTC", "ts": 1700086400, "close": 51000.0},
        ],
    )
    df = get_prices(since_ts=1700050000)
    assert len(df) == 1
    assert df.iloc[0]["close"] == 51000.0


def test_get_prices_adds_timestamp_column(db_path):
    seed_prices(db_path, [{"ticker": "BTC", "ts": 1700000000, "close": 50000.0}])
    df = get_prices()
    assert "timestamp" in df.columns


def test_safe_float_handles_edge_cases():
    assert _f(None) is None
    assert _f(float("nan")) is None
    assert _f(50.5) == 50.5
    assert _f(0) == 0.0
    assert _f("not a number") is None
