"""
Sentiment Scalper - Bull/Bear Index
------------------------------------
Composite 0-100 score that summarizes news sentiment for a given window.

Inspired by the Crypto Fear & Greed Index. Computed entirely from data
the rest of the pipeline already has — no new sources, no new fetches.

Formula:
    sentiment_score = 50 + 50 * tanh(avg_compound * 3)
    volume_score    = 50 + 50 * (bullish - bearish) / max(bullish + bearish, 1)
    index           = (sentiment_score + volume_score) / 2

Bands (match the Binance Fear & Greed convention):
    0-25   Extreme Fear   (red)
    25-45  Fear           (orange)
    45-55  Neutral        (yellow)
    55-75  Greed          (light green)
    75-100 Extreme Greed  (green)
"""

from __future__ import annotations

import math
from datetime import datetime, timedelta, timezone

from sqlalchemy import text

from db import get_engine

BULL_THRESHOLD = 0.2
BEAR_THRESHOLD = -0.2

_EMPTY: dict = {
    "score": None,
    "sentiment_score": None,
    "volume_score": None,
    "n": 0,
    "bullish": 0,
    "bearish": 0,
}


def compute_bull_bear_index(
    hours: int = 24,
    model: str | None = None,
    ago_hours: int = 0,
) -> dict:
    """
    Return the bull/bear index for a window of length `hours`, ending
    `ago_hours` hours before now.

    Examples:
        compute_bull_bear_index(24, ago_hours=0)   -> last 24h ("now")
        compute_bull_bear_index(24, ago_hours=24)  -> 24-48h ago ("yesterday")
        compute_bull_bear_index(24, ago_hours=168) -> ~1 week ago

    Returns a dict with score, sentiment_score, volume_score, n, bullish,
    bearish. score is None when the window has no mentions.
    """
    now = datetime.now(timezone.utc)
    end_ts = int((now - timedelta(hours=ago_hours)).timestamp())
    start_ts = int((now - timedelta(hours=hours + ago_hours)).timestamp())

    where = ["created_utc >= :start_ts", "created_utc < :end_ts"]
    params: dict = {"start_ts": start_ts, "end_ts": end_ts}
    if model is not None:
        where.append("model = :model")
        params["model"] = model

    sql = (
        "SELECT AVG(compound) AS avg_compound, "
        "       SUM(CASE WHEN compound >  :bull THEN 1 ELSE 0 END) AS bullish, "
        "       SUM(CASE WHEN compound < :bear THEN 1 ELSE 0 END) AS bearish, "
        "       COUNT(*) AS n "
        "FROM mentions WHERE " + " AND ".join(where)
    )
    params["bull"] = BULL_THRESHOLD
    params["bear"] = BEAR_THRESHOLD

    with get_engine().connect() as conn:
        row = conn.execute(text(sql), params).fetchone()

    if row is None or not row.n:
        return dict(_EMPTY)

    avg_compound = float(row.avg_compound or 0.0)
    bullish = int(row.bullish or 0)
    bearish = int(row.bearish or 0)
    total = bullish + bearish

    sentiment_score = 50 + 50 * math.tanh(avg_compound * 3)
    volume_score = 50 + 50 * (bullish - bearish) / total if total > 0 else 50.0
    score = (sentiment_score + volume_score) / 2

    return {
        "score": round(score, 1),
        "sentiment_score": round(sentiment_score, 1),
        "volume_score": round(volume_score, 1),
        "n": int(row.n),
        "bullish": bullish,
        "bearish": bearish,
    }


def label_for(score: float | None) -> str:
    if score is None:
        return "—"
    if score < 25:
        return "Extreme Fear"
    if score < 45:
        return "Fear"
    if score < 55:
        return "Neutral"
    if score < 75:
        return "Greed"
    return "Extreme Greed"


def color_for(score: float | None) -> str:
    """Return the band color for a given score (matches the gauge's steps)."""
    if score is None:
        return "#888888"
    if score < 25:
        return "#d62728"
    if score < 45:
        return "#f58518"
    if score < 55:
        return "#bcbd22"
    if score < 75:
        return "#9bcd9b"
    return "#2ca02c"
