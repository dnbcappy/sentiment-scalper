"""
Sentiment Scalper - v3.1
------------------------
Real data via NewsAPI (covers both stocks and crypto - indexes Cointelegraph,
CoinDesk, Decrypt, Reuters, Bloomberg, CNBC, etc.).

Pluggable sentiment engine: VADER (fast, default) or FinBERT (finance-tuned).

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
import os
import re
import sqlite3
import time
from datetime import datetime, timedelta, timezone

import requests

try:
    from dotenv import load_dotenv
    load_dotenv()
except ImportError:
    pass

# ---------- Config ----------

NEWSAPI_KEY     = os.getenv("NEWSAPI_KEY", "")
SENTIMENT_MODEL = os.getenv("SENTIMENT_MODEL", "vader").lower()

LOOKBACK_HOURS = 168
MAX_TEXT_LEN   = 2000

TICKERS = {
    "BTC":  [r"\bBTC\b", r"\bbitcoin\b"],
    "ETH":  [r"\bETH\b", r"\bethereum\b"],
    "USDT": [r"\bUSDT\b", r"\btether\b"],
    "USDC": [r"\bUSDC\b"],
    "SPY":  [r"\bSPY\b", r"\bS&P\s?500\b", r"\bS&P\b"],
    "AAPL": [r"\bAAPL\b", r"\bApple\b"],
    "TSLA": [r"\bTSLA\b", r"\bTesla\b"],
    "NVDA": [r"\bNVDA\b", r"\bNvidia\b"],
    "MSFT": [r"\bMSFT\b", r"\bMicrosoft\b"],
}

# NewsAPI search queries - one per ticker. Free tier is 100 req/day,
# so 9 tickers = ~11 runs/day max. Trim this list if you want more frequent runs.
NEWSAPI_QUERIES = {
    "BTC":  "bitcoin OR BTC",
    "ETH":  "ethereum OR ETH",
    "USDT": "tether OR USDT",
    "USDC": "USDC stablecoin",
    "SPY":  '"S&P 500" OR "SPY ETF"',
    "AAPL": "Apple stock OR AAPL",
    "TSLA": "Tesla stock OR TSLA",
    "NVDA": "Nvidia stock OR NVDA",
    "MSFT": "Microsoft stock OR MSFT",
}

DB_PATH = os.getenv(
    "DB_PATH",
    os.path.join(os.path.dirname(os.path.abspath(__file__)), "sentiment.db"),
)

_compiled = {t: [re.compile(p, re.IGNORECASE) for p in pats] for t, pats in TICKERS.items()}

def find_tickers(text: str) -> set[str]:
    if not text:
        return set()
    return {t for t, pats in _compiled.items() if any(p.search(text) for p in pats)}

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
        print("[finbert] loading model (first run downloads ~440MB)...")
        from transformers import AutoTokenizer, AutoModelForSequenceClassification
        import torch
        self.torch = torch
        self.tokenizer = AutoTokenizer.from_pretrained("ProsusAI/finbert")
        self.model = AutoModelForSequenceClassification.from_pretrained("ProsusAI/finbert")
        self.model.eval()
        print("[finbert] ready.")

    def score_batch(self, texts: list[str], batch_size: int = 16) -> list[dict]:
        out = []
        for i in range(0, len(texts), batch_size):
            chunk = [(t or "")[:MAX_TEXT_LEN] for t in texts[i:i + batch_size]]
            enc = self.tokenizer(
                chunk, padding=True, truncation=True,
                max_length=512, return_tensors="pt",
            )
            with self.torch.no_grad():
                probs = self.torch.softmax(self.model(**enc).logits, dim=-1).cpu().numpy()
            for p in probs:
                pos, neg, neu = float(p[0]), float(p[1]), float(p[2])
                out.append({"compound": pos - neg, "pos": pos, "neg": neg, "neu": neu})
        return out


def get_engine(name: str):
    if name == "vader":   return VaderEngine()
    if name == "finbert": return FinBertEngine()
    raise ValueError(f"Unknown SENTIMENT_MODEL: {name}")

# ---------- DB ----------

def init_db() -> sqlite3.Connection:
    conn = sqlite3.connect(DB_PATH)
    conn.execute("""
        CREATE TABLE IF NOT EXISTS mentions (
            id          TEXT PRIMARY KEY,
            ticker      TEXT NOT NULL,
            source      TEXT NOT NULL,
            subreddit   TEXT,           -- repurposed: publisher name
            text        TEXT,
            score       INTEGER,
            model       TEXT,
            compound    REAL,
            pos REAL, neg REAL, neu REAL,
            created_utc INTEGER NOT NULL,
            fetched_at  INTEGER NOT NULL
        )
    """)
    cols = {row[1] for row in conn.execute("PRAGMA table_info(mentions)").fetchall()}
    if "model" not in cols:
        conn.execute("ALTER TABLE mentions ADD COLUMN model TEXT")
    conn.execute("CREATE INDEX IF NOT EXISTS idx_ticker_time ON mentions(ticker, created_utc)")
    conn.commit()
    return conn

# ---------- Fetcher ----------

def fetch_newsapi() -> list[dict]:
    """Pull recent articles from NewsAPI, one query per ticker."""
    if not NEWSAPI_KEY:
        print("[newsapi] no key set in .env, aborting")
        return []

    items = []
    seen_urls = set()
    from_time = (datetime.now(timezone.utc) - timedelta(hours=LOOKBACK_HOURS)).strftime("%Y-%m-%dT%H:%M:%S")

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
                    "apiKey": NEWSAPI_KEY,
                },
                timeout=15,
            )
            if r.status_code == 401:
                print("[newsapi] 401 Unauthorized - check your NEWSAPI_KEY in .env")
                return items
            if r.status_code == 429:
                print("[newsapi] rate limit hit (100/day on free tier), stopping")
                break
            r.raise_for_status()
            data = r.json()
            articles = data.get("articles", [])
            print(f"  [{ticker}] {len(articles)} articles")
            for art in articles:
                url = art.get("url")
                if not url or url in seen_urls:
                    continue
                seen_urls.add(url)
                title = art.get("title") or ""
                desc  = art.get("description") or ""
                published = art.get("publishedAt", "")
                try:
                    created = int(datetime.fromisoformat(
                        published.replace("Z", "+00:00")
                    ).timestamp())
                except Exception:
                    created = int(datetime.now(timezone.utc).timestamp())
                items.append({
                    "id": f"newsapi_{hashlib.md5(url.encode()).hexdigest()[:16]}",
                    "source": "newsapi",
                    "subreddit": (art.get("source") or {}).get("name", "unknown"),
                    "text": f"{title}\n\n{desc}",
                    "score": 0,
                    "created_utc": created,
                })
            time.sleep(0.5)
        except Exception as e:
            print(f"  [{ticker}] error: {e}")
    return items

# ---------- Pipeline ----------

def ingest(conn: sqlite3.Connection, engine) -> int:
    print("Fetching NewsAPI (one query per ticker)...")
    items = fetch_newsapi()
    print(f"Total: {len(items)} unique articles")

    candidates = []
    for item in items:
        tickers = find_tickers(item["text"])
        if tickers:
            item["tickers"] = tickers
            candidates.append(item)
    print(f"{len(candidates)} contain tracked tickers -> scoring with {engine.name}")

    if not candidates:
        return 0

    scores = engine.score_batch([c["text"] for c in candidates])
    now = int(datetime.now(timezone.utc).timestamp())
    inserted = 0

    for item, s in zip(candidates, scores):
        for t in item["tickers"]:
            cur = conn.execute("""
                INSERT OR IGNORE INTO mentions
                (id, ticker, source, subreddit, text, score, model,
                 compound, pos, neg, neu, created_utc, fetched_at)
                VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
            """, (
                f"{item['id']}__{t}",
                t, item["source"], item["subreddit"],
                item["text"][:MAX_TEXT_LEN], item["score"], engine.name,
                s["compound"], s["pos"], s["neg"], s["neu"],
                item["created_utc"], now,
            ))
            inserted += cur.rowcount

    conn.commit()
    return inserted

def summarize(conn: sqlite3.Connection, hours: int = 24) -> None:
    cutoff = int((datetime.now(timezone.utc) - timedelta(hours=hours)).timestamp())
    rows = conn.execute("""
        SELECT ticker,
               COUNT(*)                                   AS mentions,
               AVG(compound)                              AS avg_sent,
               SUM(CASE WHEN compound >  0.2 THEN 1 END)  AS bullish,
               SUM(CASE WHEN compound < -0.2 THEN 1 END)  AS bearish
        FROM mentions
        WHERE created_utc >= ?
        GROUP BY ticker
        ORDER BY mentions DESC
    """, (cutoff,)).fetchall()

    print(f"\n=== Sentiment - last {hours}h ===")
    print(f"{'Ticker':<8}{'Mentions':>10}{'AvgSent':>10}{'Bullish':>10}{'Bearish':>10}")
    for ticker, mentions, avg, bull, bear in rows:
        print(f"{ticker:<8}{mentions:>10}{(avg or 0):>10.3f}{(bull or 0):>10}{(bear or 0):>10}")

def main() -> None:
    if not NEWSAPI_KEY:
        print("ERROR: No NEWSAPI_KEY set in .env")
        print("Get a free key at https://newsapi.org/register")
        return
    engine = get_engine(SENTIMENT_MODEL)
    conn = init_db()
    n = ingest(conn, engine)
    print(f"\nInserted {n} new rows")
    summarize(conn, hours=LOOKBACK_HOURS)
    summarize(conn, hours=24)
    conn.close()

if __name__ == "__main__":
    main()
