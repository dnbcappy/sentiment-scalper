"""
One-shot data migration: local SQLite -> remote Postgres.

Reads from the SQLite file at ./sentiment.db (or LOCAL_DB_PATH if set) and
upserts every row into the database pointed to by DATABASE_URL.

Idempotent: re-running just no-ops on rows that already exist (ON CONFLICT
DO NOTHING for mentions, ON CONFLICT DO UPDATE for prices to capture any
updated daily bars).

Usage:
    # 1. Make sure DATABASE_URL points at the remote (Supabase) DB
    # 2. Make sure migrations/0001_initial.sql has been applied there first
    # 3. Run:
    DATABASE_URL=postgresql://... python scripts/migrate_local_to_remote.py
"""

from __future__ import annotations

import logging
import os
import sqlite3
import sys
from pathlib import Path

# Allow running from the project root: scripts/ imports modules at root
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

try:
    from dotenv import load_dotenv  # noqa: E402

    load_dotenv()
except ImportError:
    pass

from sqlalchemy import create_engine, text  # noqa: E402

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s %(levelname)-7s %(name)s: %(message)s",
    datefmt="%H:%M:%S",
)
log = logging.getLogger("migrate")

LOCAL_DB_PATH = os.getenv(
    "LOCAL_DB_PATH",
    str(Path(__file__).resolve().parent.parent / "sentiment.db"),
)

REMOTE_URL = os.getenv("DATABASE_URL", "").strip()
if not REMOTE_URL:
    log.error("DATABASE_URL is not set — aborting")
    sys.exit(1)
if REMOTE_URL.startswith("postgres://"):
    REMOTE_URL = "postgresql://" + REMOTE_URL[len("postgres://") :]
if not REMOTE_URL.startswith("postgresql://"):
    log.error("DATABASE_URL doesn't look like a Postgres URL — refusing to run")
    sys.exit(1)


_INSERT_MENTION = text("""
    INSERT INTO mentions
    (id, ticker, source, subreddit, text, score, model,
     compound, pos, neg, neu, created_utc, fetched_at)
    VALUES (:id, :ticker, :source, :subreddit, :text, :score, :model,
            :compound, :pos, :neg, :neu, :created_utc, :fetched_at)
    ON CONFLICT (id) DO NOTHING
""")

_UPSERT_PRICE = text("""
    INSERT INTO prices (ticker, date, ts, open, high, low, close, volume)
    VALUES (:ticker, :date, :ts, :open, :high, :low, :close, :volume)
    ON CONFLICT (ticker, date) DO UPDATE SET
        ts     = excluded.ts,
        open   = excluded.open,
        high   = excluded.high,
        low    = excluded.low,
        close  = excluded.close,
        volume = excluded.volume
""")


def chunked(seq, size):
    for i in range(0, len(seq), size):
        yield seq[i : i + size]


def main() -> None:
    if not Path(LOCAL_DB_PATH).exists():
        log.error("Local SQLite file not found at %s", LOCAL_DB_PATH)
        sys.exit(1)

    log.info("Reading from local SQLite: %s", LOCAL_DB_PATH)
    src = sqlite3.connect(LOCAL_DB_PATH)
    src.row_factory = sqlite3.Row

    mentions = [dict(r) for r in src.execute("SELECT * FROM mentions").fetchall()]
    log.info("  %d mentions", len(mentions))

    # prices table may not exist on a brand-new DB
    try:
        prices = [dict(r) for r in src.execute("SELECT * FROM prices").fetchall()]
    except sqlite3.OperationalError:
        prices = []
    log.info("  %d prices", len(prices))
    src.close()

    log.info("Connecting to remote Postgres: %s", REMOTE_URL.split("@")[-1])
    remote = create_engine(REMOTE_URL, future=True)

    if mentions:
        log.info("Upserting %d mentions in batches of 500", len(mentions))
        with remote.begin() as conn:
            for batch in chunked(mentions, 500):
                conn.execute(_INSERT_MENTION, batch)

    if prices:
        log.info("Upserting %d prices in batches of 500", len(prices))
        with remote.begin() as conn:
            for batch in chunked(prices, 500):
                conn.execute(_UPSERT_PRICE, batch)

    # Verify final counts
    with remote.connect() as conn:
        n_mentions = conn.execute(text("SELECT COUNT(*) FROM mentions")).scalar()
        n_prices = conn.execute(text("SELECT COUNT(*) FROM prices")).scalar()
    log.info("Remote totals: %d mentions, %d prices", n_mentions, n_prices)
    log.info("Done.")


if __name__ == "__main__":
    main()
