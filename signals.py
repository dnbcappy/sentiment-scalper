"""
Sentiment Scalper - Signal Engine
----------------------------------
Computes volume-weighted sentiment signals per ticker.

Method:
  1. Divide the 7-day history into non-overlapping 6-hour buckets.
  2. Use those buckets as the baseline distribution (mean + std) for both
     mention count and average sentiment.
  3. Compare the current (active) 6-hour bucket against the baseline via z-score.
  4. Combine into a single signal:

       signal = sign(sentiment_z) * |volume_z|   when |volume_z| > VOLUME_Z_MIN
       signal = 0                                 otherwise

  The volume gate (VOLUME_Z_MIN) ensures we only fire when mention activity is
  abnormal — raw sentiment on thin volume is noise.

Known limitation: the active bucket may be partially elapsed, which biases
  volume_z downward early in a window. Signals are most reliable at bucket end.
"""

from __future__ import annotations

import sqlite3
from datetime import datetime, timezone

import numpy as np
import pandas as pd

# ---------- Constants ----------

WINDOW_HOURS     = 6      # length of the current and baseline buckets (hours)
BASELINE_HOURS   = 168    # how far back to build the baseline (7 days)
VOLUME_Z_MIN     = 1.5    # minimum |volume_z| before a signal fires
MIN_HIST_BUCKETS = 3      # tickers with fewer historical buckets are skipped

_WINDOW_SECS = WINDOW_HOURS * 3600

_COLS = [
    "ticker", "direction", "signal",
    "volume_z", "sentiment_z", "current_count", "current_sent",
]


# ---------- Public API ----------

def compute_signals(db_path: str) -> pd.DataFrame:
    """
    Load the last BASELINE_HOURS of mentions from db_path and return a
    DataFrame of tickers ranked by absolute signal strength (descending).

    Tickers with fewer than MIN_HIST_BUCKETS completed historical buckets
    are excluded — not enough data for a stable z-score.
    """
    df = _load(db_path)
    if df.empty:
        return pd.DataFrame(columns=_COLS)

    now_ts = int(datetime.now(timezone.utc).timestamp())
    current_bucket = now_ts // _WINDOW_SECS
    df["bucket"] = df["created_utc"] // _WINDOW_SECS

    rows = []
    for ticker, grp in df.groupby("ticker"):
        row = _ticker_signal(grp, current_bucket)
        if row is not None:
            rows.append(row)

    if not rows:
        return pd.DataFrame(columns=_COLS)

    return (
        pd.DataFrame(rows)
          .sort_values("signal", key=abs, ascending=False)
          .reset_index(drop=True)
    )


# ---------- Internals ----------

def _load(db_path: str) -> pd.DataFrame:
    now_ts = int(datetime.now(timezone.utc).timestamp())
    cutoff = now_ts - BASELINE_HOURS * 3600
    conn = sqlite3.connect(db_path)
    try:
        return pd.read_sql_query(
            "SELECT ticker, created_utc, compound FROM mentions WHERE created_utc >= ?",
            conn, params=(cutoff,),
        )
    finally:
        conn.close()


def _ticker_signal(grp: pd.DataFrame, current_bucket: int) -> dict | None:
    hist = grp[grp["bucket"] < current_bucket]
    curr = grp[grp["bucket"] == current_bucket]

    bucket_stats = hist.groupby("bucket").agg(
        count=("compound", "count"),
        avg_sent=("compound", "mean"),
    )

    if len(bucket_stats) < MIN_HIST_BUCKETS:
        return None

    count_mean = bucket_stats["count"].mean()
    count_std  = bucket_stats["count"].std(ddof=1)
    sent_mean  = bucket_stats["avg_sent"].mean()
    sent_std   = bucket_stats["avg_sent"].std(ddof=1)

    current_count = len(curr)
    # No activity in current window: treat sentiment as baseline (z=0, no signal)
    current_sent  = float(curr["compound"].mean()) if not curr.empty else sent_mean

    vol_z  = (current_count - count_mean) / count_std if count_std  > 0 else 0.0
    sent_z = (current_sent  - sent_mean)  / sent_std  if sent_std   > 0 else 0.0

    if abs(vol_z) > VOLUME_Z_MIN:
        signal = float(np.sign(sent_z) * abs(vol_z))
    else:
        signal = 0.0

    ticker = grp["ticker"].iloc[0]
    return {
        "ticker":        ticker,
        "direction":     "BULL" if signal > 0 else ("BEAR" if signal < 0 else "—"),
        "signal":        round(signal, 2),
        "volume_z":      round(vol_z, 2),
        "sentiment_z":   round(sent_z, 2),
        "current_count": current_count,
        "current_sent":  round(current_sent, 3),
    }
