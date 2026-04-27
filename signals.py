"""
Sentiment Scalper - Signal Engine
----------------------------------
Computes volume-weighted sentiment signals per ticker.

Method:
  1. Divide history into non-overlapping WINDOW_HOURS-long buckets.
  2. For a target bucket, use the previous BASELINE_HOURS as the baseline
     distribution (mean + std) for both mention count and avg sentiment.
  3. Compare the target bucket against the baseline via z-score.
  4. Combine into a single signal:

       signal = sign(sentiment_z) * |volume_z|   when |volume_z| > VOLUME_Z_MIN
       signal = 0                                 otherwise

  The volume gate (VOLUME_Z_MIN) ensures we only fire when mention activity
  is abnormal — raw sentiment on thin volume is noise.

Two entry points:
  - compute_signals(db_path):              live signals (current bucket vs baseline)
  - compute_historical_signals(db_path):   walks every completed past bucket,
                                            used by the backtester

Known limitation: the live current bucket may be partially elapsed, biasing
  volume_z downward early in a window. Historical signals always evaluate
  fully-elapsed buckets, so the backtest measures "complete-bucket" signals.
"""

from __future__ import annotations

import sqlite3
from datetime import datetime, timezone

import numpy as np
import pandas as pd

# ---------- Constants ----------

WINDOW_HOURS     = 6      # length of each bucket (hours)
BASELINE_HOURS   = 168    # how far back to build the baseline (7 days)
VOLUME_Z_MIN     = 1.5    # minimum |volume_z| before a signal fires
MIN_HIST_BUCKETS = 3      # tickers with fewer historical buckets are skipped

_WINDOW_SECS  = WINDOW_HOURS * 3600
_BUCKETS_BACK = BASELINE_HOURS // WINDOW_HOURS

_LIVE_COLS = [
    "ticker", "direction", "signal",
    "volume_z", "sentiment_z", "current_count", "current_sent",
]
_HIST_COLS = ["ticker", "signal_ts", "signal", "direction"]


# ---------- Public API ----------

def compute_signals(db_path: str) -> pd.DataFrame:
    """
    Live signals: evaluate the current (active) 6-hour bucket against
    the previous BASELINE_HOURS as baseline. Returns one row per ticker
    that has enough history, ranked by absolute signal strength.
    """
    df = _load(db_path, lookback_secs=BASELINE_HOURS * 3600)
    if df.empty:
        return pd.DataFrame(columns=_LIVE_COLS)

    df["bucket"] = df["created_utc"] // _WINDOW_SECS
    now_ts = int(datetime.now(timezone.utc).timestamp())
    current_bucket = now_ts // _WINDOW_SECS

    rows = []
    for ticker, grp in df.groupby("ticker"):
        s = _signal_at_bucket(grp, current_bucket)
        if s is None:
            continue
        rows.append({
            "ticker":        ticker,
            "direction":     _direction(s["signal"]),
            "signal":        round(s["signal"], 2),
            "volume_z":      round(s["volume_z"], 2),
            "sentiment_z":   round(s["sentiment_z"], 2),
            "current_count": s["current_count"],
            "current_sent":  round(s["current_sent"], 3),
        })

    if not rows:
        return pd.DataFrame(columns=_LIVE_COLS)

    return (
        pd.DataFrame(rows)
          .sort_values("signal", key=abs, ascending=False)
          .reset_index(drop=True)
    )


def compute_historical_signals(db_path: str,
                                threshold: float = VOLUME_Z_MIN) -> pd.DataFrame:
    """
    Walk every completed past bucket per ticker and emit signals where
    |signal| >= threshold. Each row is a (ticker, end-of-bucket timestamp,
    signal, direction) tuple suitable for joining against price data.

    Used by the backtester. The active bucket is excluded.
    """
    # Load all available data (no time cutoff) — historical backtest needs it
    df = _load(db_path, lookback_secs=None)
    if df.empty:
        return pd.DataFrame(columns=_HIST_COLS)

    df["bucket"] = df["created_utc"] // _WINDOW_SECS
    now_ts = int(datetime.now(timezone.utc).timestamp())
    current_bucket = now_ts // _WINDOW_SECS

    rows = []
    for ticker, grp in df.groupby("ticker"):
        for b in sorted(grp["bucket"].unique()):
            if b >= current_bucket:
                continue  # skip active/partial bucket
            s = _signal_at_bucket(grp, int(b))
            if s is None or abs(s["signal"]) < threshold:
                continue
            rows.append({
                "ticker":    ticker,
                "signal_ts": (int(b) + 1) * _WINDOW_SECS,
                "signal":    round(s["signal"], 2),
                "direction": _direction(s["signal"]),
            })

    return pd.DataFrame(rows, columns=_HIST_COLS)


# ---------- Internals ----------

def _load(db_path: str, lookback_secs: int | None) -> pd.DataFrame:
    conn = sqlite3.connect(db_path)
    try:
        if lookback_secs is None:
            return pd.read_sql_query(
                "SELECT ticker, created_utc, compound FROM mentions",
                conn,
            )
        cutoff = int(datetime.now(timezone.utc).timestamp()) - lookback_secs
        return pd.read_sql_query(
            "SELECT ticker, created_utc, compound FROM mentions WHERE created_utc >= ?",
            conn, params=(cutoff,),
        )
    finally:
        conn.close()


def _signal_at_bucket(grp: pd.DataFrame, target_bucket: int) -> dict | None:
    """
    Compute the signal as if `target_bucket` is the active bucket.
    Baseline = the BASELINE_HOURS immediately preceding it.
    Returns None if there isn't enough baseline history.
    """
    baseline_start = target_bucket - _BUCKETS_BACK
    hist = grp[(grp["bucket"] >= baseline_start) & (grp["bucket"] < target_bucket)]
    curr = grp[grp["bucket"] == target_bucket]

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
    # No activity in target bucket: sentiment defaults to baseline mean (z=0)
    current_sent  = float(curr["compound"].mean()) if not curr.empty else sent_mean

    vol_z  = (current_count - count_mean) / count_std if count_std > 0 else 0.0
    sent_z = (current_sent  - sent_mean)  / sent_std  if sent_std  > 0 else 0.0

    signal = float(np.sign(sent_z) * abs(vol_z)) if abs(vol_z) > VOLUME_Z_MIN else 0.0

    return {
        "signal":        signal,
        "volume_z":      vol_z,
        "sentiment_z":   sent_z,
        "current_count": current_count,
        "current_sent":  current_sent,
    }


def _direction(signal: float) -> str:
    if signal > 0: return "BULL"
    if signal < 0: return "BEAR"
    return "—"
