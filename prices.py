"""
Sentiment Scalper - Price Cache
--------------------------------
Pulls daily OHLC for tracked tickers via yfinance and caches them in the
configured database (SQLite locally, Postgres in production — same code).

Crypto tickers are mapped to yfinance's USD-pair symbols (BTC -> BTC-USD).
Stocks pass through unchanged. The cache uses (ticker, date) as the primary
key with INSERT ... ON CONFLICT DO UPDATE, so re-running update_prices is
idempotent and also captures intraday updates of today's bar.
"""

from __future__ import annotations

from datetime import datetime, timedelta, timezone

import pandas as pd
import yfinance as yf
from sqlalchemy import text

from db import get_engine

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

_CREATE_PRICES = """
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
"""
_CREATE_PRICES_INDEX = "CREATE INDEX IF NOT EXISTS idx_prices_ticker_ts ON prices(ticker, ts)"


def init_prices_table() -> None:
    engine = get_engine()
    with engine.begin() as conn:
        conn.execute(text(_CREATE_PRICES))
        conn.execute(text(_CREATE_PRICES_INDEX))


# ---------- Public API ----------

# ON CONFLICT DO UPDATE keeps the row fresh for today's still-changing bar.
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


def update_prices(
    tickers: list[str] | None = None, lookback_days: int = DEFAULT_LOOKBACK_DAYS
) -> int:
    """Fetch daily OHLC for the given tickers and upsert into the prices table.
    Returns the number of rows written (inserts + updates)."""
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

    init_prices_table()
    engine = get_engine()
    with engine.begin() as conn:
        result = conn.execute(_UPSERT_PRICE, rows)
    # rowcount on upserts is well-defined in both SQLite (3.24+) and Postgres
    return result.rowcount or len(rows)


def get_prices(ticker: str | None = None, since_ts: int | None = None) -> pd.DataFrame:
    """Read cached prices. Adds a 'timestamp' column for plotting convenience.
    If the prices table doesn't exist yet, returns an empty (typed) DataFrame."""
    where: list[str] = []
    params: dict = {}
    if ticker is not None:
        where.append("ticker = :ticker")
        params["ticker"] = ticker
    if since_ts is not None:
        where.append("ts >= :since_ts")
        params["since_ts"] = since_ts

    sql = "SELECT * FROM prices"
    if where:
        sql += " WHERE " + " AND ".join(where)
    sql += " ORDER BY ticker, ts"

    empty_cols = ["ticker", "date", "ts", "open", "high", "low", "close", "volume", "timestamp"]
    engine = get_engine()
    try:
        with engine.connect() as conn:
            df = pd.read_sql_query(text(sql), conn, params=params)
    except Exception:
        # Table doesn't exist yet (first run before update_prices was called)
        return pd.DataFrame(columns=empty_cols)

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
            yield _row_dict(ticker, date, row)
        return

    if not isinstance(raw.columns, pd.MultiIndex):
        return
    available = set(raw.columns.get_level_values(0))
    for sym in yf_symbols:
        if sym not in available:
            continue
        ticker = yf_to_ticker[sym]
        sub = raw[sym].dropna(how="all")
        for date, row in sub.iterrows():
            yield _row_dict(ticker, date, row)


def _row_dict(ticker: str, date, row) -> dict:
    ts = pd.Timestamp(date)
    if ts.tzinfo is None:
        ts = ts.tz_localize("UTC")
    return {
        "ticker": ticker,
        "date": ts.strftime("%Y-%m-%d"),
        "ts": int(ts.timestamp()),
        "open": _f(row.get("Open")),
        "high": _f(row.get("High")),
        "low": _f(row.get("Low")),
        "close": _f(row.get("Close")),
        "volume": _f(row.get("Volume")),
    }


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

    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s %(levelname)-7s %(name)s: %(message)s",
        datefmt="%H:%M:%S",
    )
    log = logging.getLogger(__name__)
    log.info("Updating prices...")
    n = update_prices()
    log.info("Upserted %d rows", n)
