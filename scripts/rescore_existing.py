"""
One-shot script: re-score existing VADER mentions with FinBERT.

Pulls every row in the mentions table where model='vader', scores each
unique article text with FinBERT once (cached so duplicate articles
matched to multiple tickers aren't scored twice), and inserts a new row
per (article, ticker) combination with model='finbert' and a __finbert
ID suffix.

Idempotent: skips rows where the corresponding FinBERT row already exists
(checked by ID).

Usage:
    venv/Scripts/python scripts/rescore_existing.py

Requires DATABASE_URL set (loaded from .env) and torch+transformers installed.
First run downloads ProsusAI/finbert (~440MB), cached after that.
"""

from __future__ import annotations

import logging
import sys
import time
from datetime import datetime, timezone
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

try:
    from dotenv import load_dotenv  # noqa: E402

    load_dotenv()
except ImportError:
    pass

from sqlalchemy import text  # noqa: E402

from db import get_engine  # noqa: E402
from sentiment_scalper import FinBertEngine  # noqa: E402

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s %(levelname)-7s %(name)s: %(message)s",
    datefmt="%H:%M:%S",
)
log = logging.getLogger("rescore")

BATCH_INSERT_SIZE = 500
FINBERT_BATCH_SIZE = 16

_INSERT_MENTION = text("""
    INSERT INTO mentions
    (id, ticker, source, subreddit, text, score, model,
     compound, pos, neg, neu, created_utc, fetched_at)
    VALUES (:id, :ticker, :source, :subreddit, :text, :score, :model,
            :compound, :pos, :neg, :neu, :created_utc, :fetched_at)
    ON CONFLICT (id) DO NOTHING
""")


def main() -> None:
    engine = get_engine()

    # 1. Load every VADER row
    log.info("Loading VADER rows from database...")
    with engine.connect() as conn:
        vader_rows = conn.execute(
            text("""
                SELECT id, ticker, source, subreddit, text, score,
                       created_utc, fetched_at
                FROM mentions
                WHERE model = 'vader'
            """)
        ).fetchall()
    log.info("Loaded %d VADER rows", len(vader_rows))

    # 2. Find which FinBERT rows already exist (so we can skip them)
    with engine.connect() as conn:
        existing_finbert = {
            r[0]
            for r in conn.execute(
                text("SELECT id FROM mentions WHERE model = 'finbert'")
            ).fetchall()
        }
    log.info("FinBERT rows already in DB: %d", len(existing_finbert))

    # 3. Build the list of rows that still need a FinBERT counterpart
    to_score: list[dict] = []
    for vader_id, ticker, source, subreddit, text_val, score, created_utc, fetched_at in vader_rows:
        finbert_id = vader_id.replace("__vader", "__finbert")
        if finbert_id in existing_finbert:
            continue
        to_score.append(
            {
                "finbert_id": finbert_id,
                "ticker": ticker,
                "source": source,
                "subreddit": subreddit,
                "text": text_val,
                "score": score,
                "created_utc": created_utc,
                "fetched_at": fetched_at,
            }
        )

    if not to_score:
        log.info("Nothing to do — all VADER rows already have FinBERT counterparts.")
        return
    log.info("Rows needing a FinBERT score: %d", len(to_score))

    # 4. Collect unique article texts — score each only once
    unique_texts = sorted({r["text"] for r in to_score})
    log.info(
        "Unique article texts to score: %d (saving %d duplicate scorings)",
        len(unique_texts),
        len(to_score) - len(unique_texts),
    )

    # 5. Instantiate FinBERT (this triggers the model download on first run)
    log.info("Loading FinBERT model...")
    t0 = time.time()
    finbert = FinBertEngine()
    log.info("FinBERT loaded in %.1fs", time.time() - t0)

    # 6. Score in batches
    log.info("Scoring %d texts in batches of %d...", len(unique_texts), FINBERT_BATCH_SIZE)
    t0 = time.time()
    scores = finbert.score_batch(unique_texts, batch_size=FINBERT_BATCH_SIZE)
    log.info(
        "Scored in %.1fs (%.2fs per text avg)",
        time.time() - t0,
        (time.time() - t0) / len(unique_texts),
    )

    # 7. Build the score lookup table
    score_by_text = dict(zip(unique_texts, scores, strict=True))

    # 8. Build the insert rows
    now = int(datetime.now(timezone.utc).timestamp())
    insert_rows = []
    for r in to_score:
        s = score_by_text[r["text"]]
        insert_rows.append(
            {
                "id": r["finbert_id"],
                "ticker": r["ticker"],
                "source": r["source"],
                "subreddit": r["subreddit"],
                "text": r["text"],
                "score": r["score"],
                "model": "finbert",
                "compound": s["compound"],
                "pos": s["pos"],
                "neg": s["neg"],
                "neu": s["neu"],
                "created_utc": r["created_utc"],
                "fetched_at": now,
            }
        )

    # 9. Batched insert
    log.info("Inserting %d FinBERT rows in batches of %d...", len(insert_rows), BATCH_INSERT_SIZE)
    with engine.begin() as conn:
        for i in range(0, len(insert_rows), BATCH_INSERT_SIZE):
            conn.execute(_INSERT_MENTION, insert_rows[i : i + BATCH_INSERT_SIZE])

    # 10. Verify totals
    with engine.connect() as conn:
        n_vader = conn.execute(text("SELECT COUNT(*) FROM mentions WHERE model='vader'")).scalar()
        n_finbert = conn.execute(
            text("SELECT COUNT(*) FROM mentions WHERE model='finbert'")
        ).scalar()
    log.info("Done. Final counts: VADER=%d, FinBERT=%d", n_vader, n_finbert)


if __name__ == "__main__":
    main()
