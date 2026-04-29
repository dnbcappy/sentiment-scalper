"""
Sentiment Scalper - Price Cache
--------------------------------
Pulls daily OHLC for tracked tickers via yfinance and caches them in the
same SQLite DB used for mentions.

Crypto tickers are mapped to yfinance's USD-pair symbols (BTC -> BTC-USD).
Stocks pass through unchanged. The cache uses (ticker, date) as the primary
key with INSERT OR REPLACE, so re-running update_prices is idempotent and
also captures intraday updates of today's bar.
"""

from __future__ import annotations

import sqlite3
from datetime import datetime, timedelta, timezone

import pandas as pd
import yfinance as yf

# ---------- Symbol mapping ----------

YF_SYMBOL = {
    "BTC": "BTC-USD",
    "ETH": "ETH-USD",
    "USDT": "USDT-USD",
    "USDC": "USDC-USD",
    "SPY": "SPY",
    "AAPL": "AAPL",
    "TSLA": "TSLA",
    "NVDA": "NVDA",
    "MSFT": "MSFT",
}

DEFAULT_LOOKBACK_DAYS = 60


# ---------- Schema ----------


def init_prices_table(conn: sqlite3.Connection) -> None:
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
    conn.execute("CREATE INDEX IF NOT EXISTS idx_prices_ticker_ts ON prices(ticker, ts)")
    conn.commit()


# ---------- Public API ----------


def update_prices(
    db_path: str, tickers: list[str] | None = None, lookback_days: int = DEFAULT_LOOKBACK_DAYS
) -> int:
    """
    Fetch daily OHLC for the given tickers and upsert into the prices table.
    Returns the number of rows touched.
    """
    if tickers is None:
        tickers = list(YF_SYMBOL.keys())

    yf_to_ticker = {YF_SYMBOL[t]: t for t in tickers if t in YF_SYMBOL}
    if not yf_to_ticker:
        return 0

    end = datetime.now(timezone.utc)
    start = end - timedelta(days=lookback_days)

    raw = yf.download(
        list(yf_to_ticker.keys()),
        start=start.strftime("%Y-%m-%d"),
        end=(end + timedelta(days=1)).strftime("%Y-%m-%d"),
        progress=False,
        auto_adjust=False,
        group_by="ticker",
    )

    if raw is None or raw.empty:
        return 0

    rows = list(_iter_rows(raw, yf_to_ticker))
    if not rows:
        return 0

    conn = sqlite3.connect(db_path)
    try:
        init_prices_table(conn)
        cur = conn.executemany(
            """
            INSERT OR REPLACE INTO prices
            (ticker, date, ts, open, high, low, close, volume)
            VALUES (?, ?, ?, ?, ?, ?, ?, ?)
        """,
            rows,
        )
        conn.commit()
        return cur.rowcount
    finally:
        conn.close()


def get_prices(
    db_path: str, ticker: str | None = None, since_ts: int | None = None
) -> pd.DataFrame:
    """
    Read cached prices. If ticker is None, returns all tickers.
    Adds a 'timestamp' column (UTC datetime) for plotting convenience.
    """
    sql = "SELECT * FROM prices"
    where = []
    params: list = []
    if ticker is not None:
        where.append("ticker = ?")
        params.append(ticker)
    if since_ts is not None:
        where.append("ts >= ?")
        params.append(since_ts)
    if where:
        sql += " WHERE " + " AND ".join(where)
    sql += " ORDER BY ticker, ts"

    conn = sqlite3.connect(db_path)
    try:
        df = pd.read_sql_query(sql, conn, params=params)
    except pd.io.sql.DatabaseError:
        # Table doesn't exist yet — first run before update_prices was called
        return pd.DataFrame(
            columns=["ticker", "date", "ts", "open", "high", "low", "close", "volume", "timestamp"]
        )
    finally:
        conn.close()

    if not df.empty:
        df["timestamp"] = pd.to_datetime(df["ts"], unit="s", utc=True)
    return df


# ---------- Internals ----------


def _iter_rows(raw: pd.DataFrame, yf_to_ticker: dict[str, str]):
    """Flatten the yfinance frame (single or multi-ticker) into DB rows."""
    yf_symbols = list(yf_to_ticker.keys())

    if len(yf_symbols) == 1:
        sym = yf_symbols[0]
        ticker = yf_to_ticker[sym]
        for date, row in raw.dropna(how="all").iterrows():
            yield _row_tuple(ticker, date, row)
        return

    # Multi-ticker: MultiIndex columns. yfinance can put ticker as level 0 or 1
    # depending on group_by; we requested group_by='ticker' so ticker is level 0.
    if not isinstance(raw.columns, pd.MultiIndex):
        return
    available = set(raw.columns.get_level_values(0))
    for sym in yf_symbols:
        if sym not in available:
            continue
        ticker = yf_to_ticker[sym]
        sub = raw[sym].dropna(how="all")
        for date, row in sub.iterrows():
            yield _row_tuple(ticker, date, row)


def _row_tuple(ticker: str, date, row) -> tuple:
    ts = pd.Timestamp(date)
    if ts.tzinfo is None:
        ts = ts.tz_localize("UTC")
    return (
        ticker,
        ts.strftime("%Y-%m-%d"),
        int(ts.timestamp()),
        _f(row.get("Open")),
        _f(row.get("High")),
        _f(row.get("Low")),
        _f(row.get("Close")),
        _f(row.get("Volume")),
    )


def _f(x) -> float | None:
    if x is None or pd.isna(x):
        return None
    try:
        return float(x)
    except (TypeError, ValueError):
        return None


# ---------- CLI entry point ----------

if __name__ == "__main__":
    import logging
    import os

    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s %(levelname)-7s %(name)s: %(message)s",
        datefmt="%H:%M:%S",
    )
    log = logging.getLogger(__name__)
    db = os.getenv(
        "DB_PATH", os.path.join(os.path.dirname(os.path.abspath(__file__)), "sentiment.db")
    )
    log.info("Updating prices into %s", db)
    n = update_prices(db)
    log.info("Upserted %d rows", n)
