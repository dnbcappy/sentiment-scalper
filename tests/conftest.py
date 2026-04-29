"""Shared fixtures and helpers for the test suite.

Strategy: each test gets its own SQLite file in a tmp_path, and DATABASE_URL
is monkeypatched to point at it. The cached engine is reset before and after
the test so each test runs in isolation.
"""

from __future__ import annotations

import sqlite3
from typing import Any

import pytest


@pytest.fixture
def db_path(tmp_path, monkeypatch) -> str:
    """A fresh sentiment.db with the production schema and no rows.

    Sets DATABASE_URL to point at this file so all production code paths
    (which use db.get_engine()) automatically use the temp DB.
    """
    path = tmp_path / "sentiment.db"
    conn = sqlite3.connect(path)
    conn.execute("""
        CREATE TABLE mentions (
            id          TEXT PRIMARY KEY,
            ticker      TEXT NOT NULL,
            source      TEXT NOT NULL,
            subreddit   TEXT,
            text        TEXT,
            score       INTEGER,
            model       TEXT,
            compound    REAL,
            pos REAL, neg REAL, neu REAL,
            created_utc INTEGER NOT NULL,
            fetched_at  INTEGER NOT NULL
        )
    """)
    conn.commit()
    conn.close()

    monkeypatch.setenv("DATABASE_URL", f"sqlite:///{path}")

    # Clear the cached engine so the new DATABASE_URL takes effect this test
    from db import reset_engine

    reset_engine()
    yield str(path)
    reset_engine()


def seed_mentions(db_path: str, rows: list[dict[str, Any]]) -> None:
    """Insert mention rows. Each row needs at least ticker, created_utc, compound.
    Optional: model (default 'vader'), id (default synthetic)."""
    conn = sqlite3.connect(db_path)
    for i, row in enumerate(rows):
        ticker = row["ticker"]
        model = row.get("model", "vader")
        row_id = row.get("id", f"test_{i}_{ticker}_{model}")
        conn.execute(
            "INSERT INTO mentions "
            "(id, ticker, source, subreddit, text, score, model, "
            " compound, pos, neg, neu, created_utc, fetched_at) "
            "VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)",
            (
                row_id,
                ticker,
                row.get("source", "newsapi"),
                row.get("subreddit", "test"),
                row.get("text", "test text"),
                0,
                model,
                row["compound"],
                0.0,
                0.0,
                0.0,
                row["created_utc"],
                0,
            ),
        )
    conn.commit()
    conn.close()


def baseline_rows(
    ticker: str, current_bucket: int, window_secs: int, model: str = "vader"
) -> list[dict[str, Any]]:
    """5 baseline buckets with varying counts and slight sentiment variation.

    Counts: [1, 2, 3, 2, 1] -> non-zero std, so z-score math actually fires.
    Sentiments: small variation around 0 -> non-zero sent_std.
    """
    counts = [1, 2, 3, 2, 1]
    sentiments = [0.0, 0.1, -0.1, 0.05, -0.05]
    rows: list[dict[str, Any]] = []
    for offset, (n, s) in enumerate(zip(counts, sentiments, strict=True), start=1):
        bucket_ts = (current_bucket - offset) * window_secs + 100
        for _ in range(n):
            rows.append({"ticker": ticker, "created_utc": bucket_ts, "compound": s, "model": model})
    return rows


def seed_prices(db_path: str, rows: list[dict[str, Any]]) -> None:
    """Insert daily price rows. Each row needs ticker, ts, close. Other OHLCV optional."""
    conn = sqlite3.connect(db_path)
    conn.execute("""
        CREATE TABLE IF NOT EXISTS prices (
            ticker  TEXT NOT NULL,
            date    TEXT NOT NULL,
            ts      INTEGER NOT NULL,
            open    REAL,
            high    REAL,
            low     REAL,
            close   REAL,
            volume  REAL,
            PRIMARY KEY (ticker, date)
        )
    """)
    for row in rows:
        from datetime import datetime, timezone

        date = datetime.fromtimestamp(row["ts"], tz=timezone.utc).strftime("%Y-%m-%d")
        conn.execute(
            "INSERT OR REPLACE INTO prices "
            "(ticker, date, ts, open, high, low, close, volume) "
            "VALUES (?, ?, ?, ?, ?, ?, ?, ?)",
            (
                row["ticker"],
                date,
                row["ts"],
                row.get("open", row["close"]),
                row.get("high", row["close"]),
                row.get("low", row["close"]),
                row["close"],
                row.get("volume", 0.0),
            ),
        )
    conn.commit()
    conn.close()
