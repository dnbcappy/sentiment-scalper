"""
Sentiment Scalper - v3.2
------------------------
Real data via NewsAPI (covers both stocks and crypto - indexes Cointelegraph,
CoinDesk, Decrypt, Reuters, Bloomberg, CNBC, etc.).

Pluggable sentiment engine: VADER (fast, default) or FinBERT (finance-tuned).

Backend: SQLite locally by default. Set DATABASE_URL=postgresql://... to point
at a hosted Postgres (Supabase, Cloud SQL, etc.) — same code path either way.

Setup:
    pip install -r requirements.txt
    cp .env.example .env  # then fill in NEWSAPI_KEY

Get a NewsAPI key (free, email signup):
    https://newsapi.org/register

Run:
    python sentiment_scalper.py
"""

from __future__ import annotations

import hashlib
import logging
import os
import re
import time
from datetime import datetime, timedelta, timezone

import requests
from sqlalchemy import inspect, text

from db import get_engine

try:
    from dotenv import load_dotenv

    load_dotenv()
except ImportError:
    pass

logger = logging.getLogger(__name__)

# ---------- Config ----------

NEWSAPI_KEY = os.getenv("NEWSAPI_KEY", "")
SENTIMENT_MODEL = os.getenv("SENTIMENT_MODEL", "vader").lower()

LOOKBACK_HOURS = 168
MAX_TEXT_LEN = 2000

# Publisher whitelist for NewsAPI. Without this, queries pull in random
# blogs / press releases / PyPI announcements that mention "Apple" or
# "Bitcoin" without being financial news, polluting the sentiment signal.
NEWSAPI_DOMAINS = ",".join(
    [
        # Stocks / general finance
        "bloomberg.com",
        "reuters.com",
        "cnbc.com",
        "wsj.com",
        "ft.com",
        "marketwatch.com",
        "barrons.com",
        "businessinsider.com",
        "fortune.com",
        "forbes.com",
        "nasdaq.com",
        # Crypto
        "cointelegraph.com",
        "coindesk.com",
        "decrypt.co",
        "theblock.co",
        "cryptobriefing.com",
        "bitcoinmagazine.com",
    ]
)

TICKERS = {
    "BTC": [r"\bBTC\b", r"\bbitcoin\b"],
    "ETH": [r"\bETH\b", r"\bethereum\b"],
    "USDT": [r"\bUSDT\b", r"\btether\b"],
    "USDC": [r"\bUSDC\b"],
    "SPY": [r"\bSPY\b", r"\bS&P\s?500\b", r"\bS&P\b"],
    "AAPL": [r"\bAAPL\b", r"\bApple\b"],
    "TSLA": [r"\bTSLA\b", r"\bTesla\b"],
    "NVDA": [r"\bNVDA\b", r"\bNvidia\b"],
    "MSFT": [r"\bMSFT\b", r"\bMicrosoft\b"],
}

# NewsAPI search queries - one per ticker. Free tier is 100 req/day,
# so 9 tickers = ~11 runs/day max. Trim this list if you want more frequent runs.
NEWSAPI_QUERIES = {
    "BTC": "bitcoin OR BTC",
    "ETH": "ethereum OR ETH",
    "USDT": "tether OR USDT",
    "USDC": "USDC stablecoin",
    "SPY": '"S&P 500" OR "SPY ETF"',
    "AAPL": "Apple stock OR AAPL",
    "TSLA": "Tesla stock OR TSLA",
    "NVDA": "Nvidia stock OR NVDA",
    "MSFT": "Microsoft stock OR MSFT",
}

_compiled = {t: [re.compile(p, re.IGNORECASE) for p in pats] for t, pats in TICKERS.items()}


def find_tickers(text_input: str) -> set[str]:
    if not text_input:
        return set()
    return {t for t, pats in _compiled.items() if any(p.search(text_input) for p in pats)}


# ---------- Sentiment engines ----------


class VaderEngine:
    """Fast, rule-based. Good baseline. Misses finance jargon."""

    name = "vader"

    def __init__(self):
        from vaderSentiment.vaderSentiment import SentimentIntensityAnalyzer

        self._a = SentimentIntensityAnalyzer()

    def score_batch(self, texts: list[str]) -> list[dict]:
        return [self._a.polarity_scores(t or "") for t in texts]


class FinBertEngine:
    """Finance-tuned BERT. Heavier (~440MB download, slower) but understands
    'beat earnings', 'guided lower', 'rate cut', etc."""

    name = "finbert"

    def __init__(self):
        logger.info("Loading FinBERT model (first run downloads ~440MB)...")
        import torch
        from transformers import AutoModelForSequenceClassification, AutoTokenizer

        self.torch = torch
        self.tokenizer = AutoTokenizer.from_pretrained("ProsusAI/finbert")
        self.model = AutoModelForSequenceClassification.from_pretrained("ProsusAI/finbert")
        self.model.eval()
        logger.info("FinBERT ready.")

    def score_batch(self, texts: list[str], batch_size: int = 16) -> list[dict]:
        out = []
        for i in range(0, len(texts), batch_size):
            chunk = [(t or "")[:MAX_TEXT_LEN] for t in texts[i : i + batch_size]]
            enc = self.tokenizer(
                chunk,
                padding=True,
                truncation=True,
                max_length=512,
                return_tensors="pt",
            )
            with self.torch.no_grad():
                probs = self.torch.softmax(self.model(**enc).logits, dim=-1).cpu().numpy()
            for p in probs:
                pos, neg, neu = float(p[0]), float(p[1]), float(p[2])
                out.append({"compound": pos - neg, "pos": pos, "neg": neg, "neu": neu})
        return out


def get_sentiment_engine(name: str):
    if name == "vader":
        return VaderEngine()
    if name == "finbert":
        return FinBertEngine()
    raise ValueError(f"Unknown SENTIMENT_MODEL: {name}")


# ---------- DB ----------

# Schema is identical between SQLite and Postgres. SQLite 3.24+ accepts
# the standard SQL syntax for most things; the only place we diverge is
# adding the model column on existing rows where Postgres doesn't tolerate
# the silent-add-if-missing pattern, so we use proper introspection.

_CREATE_MENTIONS = """
CREATE TABLE IF NOT EXISTS mentions (
    id          TEXT PRIMARY KEY,
    ticker      TEXT NOT NULL,
    source      TEXT NOT NULL,
    subreddit   TEXT,
    text        TEXT,
    score       INTEGER,
    model       TEXT,
    compound    REAL,
    pos         REAL,
    neg         REAL,
    neu         REAL,
    created_utc INTEGER NOT NULL,
    fetched_at  INTEGER NOT NULL
)
"""

_CREATE_INDEX_TICKER_TIME = (
    "CREATE INDEX IF NOT EXISTS idx_ticker_time ON mentions(ticker, created_utc)"
)
_CREATE_INDEX_MODEL = "CREATE INDEX IF NOT EXISTS idx_model ON mentions(model)"


def init_db() -> None:
    """Create the mentions table and indexes if they don't exist, then run
    the ID migration. Safe to call repeatedly (idempotent)."""
    engine = get_engine()
    with engine.begin() as conn:
        conn.execute(text(_CREATE_MENTIONS))

        # Backfill: older databases predate the model column. Add it if
        # introspection says it's missing.
        existing = {c["name"] for c in inspect(engine).get_columns("mentions")}
        if "model" not in existing:
            conn.execute(text("ALTER TABLE mentions ADD COLUMN model TEXT"))

        conn.execute(text(_CREATE_INDEX_TICKER_TIME))
        conn.execute(text(_CREATE_INDEX_MODEL))

    _migrate_ids_v2()


def _migrate_ids_v2() -> int:
    """Suffix mention IDs with '__<model>' so the same article can carry scores
    from multiple engines without primary-key collisions. Idempotent: rows that
    already end with the model suffix are skipped."""
    engine = get_engine()
    with engine.begin() as conn:
        rows = conn.execute(
            text("SELECT id, model FROM mentions WHERE model IS NOT NULL")
        ).fetchall()
        updates = [
            {"new_id": f"{row_id}__{model}", "old_id": row_id}
            for row_id, model in rows
            if not row_id.endswith(f"__{model}")
        ]
        if updates:
            conn.execute(text("UPDATE mentions SET id = :new_id WHERE id = :old_id"), updates)
            logger.info("Migrated %d mention IDs to include model suffix", len(updates))
        return len(updates)


# ---------- Fetcher ----------


def fetch_newsapi() -> list[dict]:
    """Pull recent articles from NewsAPI, one query per ticker."""
    if not NEWSAPI_KEY:
        logger.error("No NEWSAPI_KEY set in environment, aborting fetch")
        return []

    items = []
    seen_urls = set()
    from_time = (datetime.now(timezone.utc) - timedelta(hours=LOOKBACK_HOURS)).strftime(
        "%Y-%m-%dT%H:%M:%S"
    )

    for ticker, query in NEWSAPI_QUERIES.items():
        try:
            r = requests.get(
                "https://newsapi.org/v2/everything",
                params={
                    "q": query,
                    "from": from_time,
                    "language": "en",
                    "sortBy": "publishedAt",
                    "pageSize": 100,
                    "domains": NEWSAPI_DOMAINS,
                    "apiKey": NEWSAPI_KEY,
                },
                timeout=15,
            )
            if r.status_code == 401:
                logger.error("NewsAPI 401 Unauthorized — check NEWSAPI_KEY in .env")
                return items
            if r.status_code == 429:
                logger.warning("NewsAPI rate limit hit (free tier), stopping fetch")
                break
            r.raise_for_status()
            data = r.json()
            articles = data.get("articles", [])
            logger.info("Fetched %d articles for %s", len(articles), ticker)
            for art in articles:
                url = art.get("url")
                if not url or url in seen_urls:
                    continue
                seen_urls.add(url)
                title = art.get("title") or ""
                desc = art.get("description") or ""
                published = art.get("publishedAt", "")
                try:
                    created = int(
                        datetime.fromisoformat(published.replace("Z", "+00:00")).timestamp()
                    )
                except Exception:
                    # Skip rather than fall back to now() — bad timestamps
                    # would otherwise pile every malformed article into the
                    # active bucket, distorting signal calculations.
                    logger.warning(
                        "Skipping article from %s — unparseable timestamp %r",
                        (art.get("source") or {}).get("name", "unknown"),
                        published,
                    )
                    continue
                items.append(
                    {
                        "id": f"newsapi_{hashlib.md5(url.encode()).hexdigest()[:16]}",
                        "source": "newsapi",
                        "subreddit": (art.get("source") or {}).get("name", "unknown"),
                        "text": f"{title}\n\n{desc}",
                        "score": 0,
                        "created_utc": created,
                    }
                )
            time.sleep(0.5)
        except requests.exceptions.RequestException as e:
            # Don't pass the exception object — its repr can include the
            # request URL, which has the apiKey in the query string.
            logger.error("NewsAPI request failed for %s: %s", ticker, type(e).__name__)
        except Exception as e:
            logger.error("Unexpected error processing %s: %s", ticker, type(e).__name__)
    return items


# ---------- Pipeline ----------


_INSERT_MENTION = text("""
    INSERT INTO mentions
    (id, ticker, source, subreddit, text, score, model,
     compound, pos, neg, neu, created_utc, fetched_at)
    VALUES (:id, :ticker, :source, :subreddit, :text, :score, :model,
            :compound, :pos, :neg, :neu, :created_utc, :fetched_at)
    ON CONFLICT (id) DO NOTHING
""")


def ingest(sentiment_engine) -> int:
    """Fetch, filter, score, and insert mentions. Returns the number of new rows."""
    logger.info("Fetching NewsAPI (one query per ticker)...")
    newsapi_items = fetch_newsapi()
    logger.info("NewsAPI: %d articles", len(newsapi_items))

    logger.info("Fetching RSS feeds...")
    from rss_scraper import fetch_rss

    rss_items = fetch_rss()
    logger.info("RSS: %d entries", len(rss_items))

    # Within-run dedup across sources: when NewsAPI and RSS surface the same
    # URL (likely when they cover the same publisher), the URL hash is the
    # same — extract it from the ID (format: "<source>_<hash>") and keep
    # only the first occurrence. Cross-run dedup is handled separately by
    # the ON CONFLICT DO NOTHING insert at the DB level.
    seen_hashes: set[str] = set()
    items: list[dict] = []
    for item in newsapi_items + rss_items:
        parts = item["id"].split("_", 1)
        url_hash = parts[1] if len(parts) == 2 else item["id"]
        if url_hash in seen_hashes:
            continue
        seen_hashes.add(url_hash)
        items.append(item)
    deduped = (len(newsapi_items) + len(rss_items)) - len(items)
    if deduped > 0:
        logger.info("Deduped %d cross-source URL overlaps", deduped)
    logger.info("Total unique items this run: %d", len(items))

    candidates = []
    for item in items:
        tickers = find_tickers(item["text"])
        if tickers:
            item["tickers"] = tickers
            candidates.append(item)
    logger.info(
        "%d items contain tracked tickers — scoring with %s",
        len(candidates),
        sentiment_engine.name,
    )

    if not candidates:
        return 0

    scores = sentiment_engine.score_batch([c["text"] for c in candidates])
    now = int(datetime.now(timezone.utc).timestamp())

    rows = []
    for item, s in zip(candidates, scores, strict=True):
        for t in item["tickers"]:
            rows.append(
                {
                    "id": f"{item['id']}__{t}__{sentiment_engine.name}",
                    "ticker": t,
                    "source": item["source"],
                    "subreddit": item["subreddit"],
                    "text": item["text"][:MAX_TEXT_LEN],
                    "score": item["score"],
                    "model": sentiment_engine.name,
                    "compound": s["compound"],
                    "pos": s["pos"],
                    "neg": s["neg"],
                    "neu": s["neu"],
                    "created_utc": item["created_utc"],
                    "fetched_at": now,
                }
            )

    if not rows:
        return 0

    engine = get_engine()
    with engine.begin() as conn:
        result = conn.execute(_INSERT_MENTION, rows)
    # rowcount is reliable for ON CONFLICT DO NOTHING in both SQLite and Postgres
    return result.rowcount or 0


def summarize(hours: int = 24) -> None:
    """Print a human-readable per-ticker summary for the last `hours`."""
    cutoff = int((datetime.now(timezone.utc) - timedelta(hours=hours)).timestamp())
    engine = get_engine()
    with engine.connect() as conn:
        rows = conn.execute(
            text("""
                SELECT ticker,
                       COUNT(*)                                    AS mentions,
                       AVG(compound)                               AS avg_sent,
                       SUM(CASE WHEN compound >  0.2 THEN 1 ELSE 0 END) AS bullish,
                       SUM(CASE WHEN compound < -0.2 THEN 1 ELSE 0 END) AS bearish
                FROM mentions
                WHERE created_utc >= :cutoff
                GROUP BY ticker
                ORDER BY mentions DESC
            """),
            {"cutoff": cutoff},
        ).fetchall()

    print(f"\n=== Sentiment - last {hours}h ===")
    print(f"{'Ticker':<8}{'Mentions':>10}{'AvgSent':>10}{'Bullish':>10}{'Bearish':>10}")
    for ticker, mentions, avg, bull, bear in rows:
        print(f"{ticker:<8}{mentions:>10}{(avg or 0):>10.3f}{(bull or 0):>10}{(bear or 0):>10}")


def main() -> None:
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s %(levelname)-7s %(name)s: %(message)s",
        datefmt="%H:%M:%S",
    )
    if not NEWSAPI_KEY:
        logger.error("No NEWSAPI_KEY set in .env. Get a free key at https://newsapi.org/register")
        return
    sentiment_engine = get_sentiment_engine(SENTIMENT_MODEL)
    init_db()
    n = ingest(sentiment_engine)
    logger.info("Inserted %d new rows", n)
    summarize(hours=LOOKBACK_HOURS)
    summarize(hours=24)


if __name__ == "__main__":
    main()
