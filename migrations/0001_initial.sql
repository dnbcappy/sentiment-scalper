-- Sentiment Scalper - initial Postgres schema
-- Run once against your hosted database (e.g., Supabase) to create the
-- tables that sentiment_scalper.py and prices.py expect.
--
-- Usage (Supabase): paste the contents into the SQL editor and run.
-- Usage (psql):     psql "$DATABASE_URL" -f migrations/0001_initial.sql
--
-- Idempotent: safe to re-run. CREATE IF NOT EXISTS guards every object.

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
    created_utc BIGINT NOT NULL,
    fetched_at  BIGINT NOT NULL
);

CREATE INDEX IF NOT EXISTS idx_ticker_time ON mentions(ticker, created_utc);
CREATE INDEX IF NOT EXISTS idx_model       ON mentions(model);


CREATE TABLE IF NOT EXISTS prices (
    ticker  TEXT NOT NULL,
    date    TEXT NOT NULL,
    ts      BIGINT NOT NULL,
    open    REAL,
    high    REAL,
    low     REAL,
    close   REAL,
    volume  REAL,
    PRIMARY KEY (ticker, date)
);

CREATE INDEX IF NOT EXISTS idx_prices_ticker_ts ON prices(ticker, ts);
