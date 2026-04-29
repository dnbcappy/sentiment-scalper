"""
Sentiment Scalper - Backtest
-----------------------------
Joins historical signals (signals.compute_historical_signals) against cached
daily prices (prices.get_prices) and computes a hit rate per ticker.

Hit definition: at horizon H days after the signal,
    BULL signal hit  ⇔  price went up  (return > 0)
    BEAR signal hit  ⇔  price went down (return < 0)

Entry/exit prices use "last close at or before timestamp" — simple and
realistic, but means weekend/holiday signals on stocks may show 0 return
at short horizons. Crypto trades 24/7, so this only affects equities.
"""

from __future__ import annotations

import pandas as pd

from prices import get_prices
from signals import compute_historical_signals

HORIZONS_DAYS = (1, 3, 7)

_OUT_COLS = (
    ["ticker", "n_signals"]
    + [f"hit_rate_{h}d" for h in HORIZONS_DAYS]
    + [f"avg_return_{h}d" for h in HORIZONS_DAYS]
)


def compute_hit_rates(db_path: str, threshold: float, model: str | None = None) -> pd.DataFrame:
    """
    Per-ticker hit rates at 1d/3d/7d horizons, plus an aggregate 'ALL' row.
    Returns an empty (but well-typed) DataFrame if there are no signals or
    no price data yet. If `model` is given, only mentions scored by that
    engine are used to compute the historical signals.
    """
    sigs = compute_historical_signals(db_path, threshold=threshold, model=model)
    if sigs.empty:
        return _empty()

    prices = get_prices(db_path)
    if prices.empty:
        return _empty()

    detail = _build_detail(sigs, prices)
    if detail.empty:
        return _empty()

    rows = [{"ticker": ticker, **_aggregate(group)} for ticker, group in detail.groupby("ticker")]
    rows.append({"ticker": "ALL", **_aggregate(detail)})

    return pd.DataFrame(rows)[_OUT_COLS]


# ---------- Internals ----------


def _build_detail(sigs: pd.DataFrame, prices: pd.DataFrame) -> pd.DataFrame:
    """One row per signal. Horizon outcomes are NaN if the horizon hasn't
    elapsed yet (exit timestamp later than our latest price bar) — treats
    those positions as 'still open' rather than reporting a spurious 0% return."""
    rows = []
    for ticker, group in sigs.groupby("ticker"):
        ticker_prices = prices[prices["ticker"] == ticker].sort_values("ts")
        if ticker_prices.empty:
            continue
        ts_array = ticker_prices["ts"].to_numpy()
        close_array = ticker_prices["close"].to_numpy()
        latest_ts = int(ts_array[-1])

        for _, sig in group.iterrows():
            entry_ts = int(sig["signal_ts"])
            entry = _close_at_or_before(ts_array, close_array, entry_ts)
            if entry is None or entry == 0:
                continue
            row = {"ticker": ticker, "direction": sig["direction"]}
            for h in HORIZONS_DAYS:
                exit_ts = entry_ts + h * 86400
                if exit_ts > latest_ts:
                    # Horizon not yet reached — position still open
                    row[f"return_{h}d"] = None
                    row[f"hit_{h}d"] = None
                    continue
                exit_close = _close_at_or_before(ts_array, close_array, exit_ts)
                if exit_close is None:
                    row[f"return_{h}d"] = None
                    row[f"hit_{h}d"] = None
                    continue
                ret = (exit_close - entry) / entry
                row[f"return_{h}d"] = ret
                if sig["direction"] == "BULL":
                    row[f"hit_{h}d"] = ret > 0
                else:
                    row[f"hit_{h}d"] = ret < 0
            rows.append(row)
    return pd.DataFrame(rows)


def _aggregate(detail: pd.DataFrame) -> dict:
    out = {"n_signals": int(len(detail))}
    for h in HORIZONS_DAYS:
        hits = detail[f"hit_{h}d"].dropna()
        rets = detail[f"return_{h}d"].dropna()
        out[f"hit_rate_{h}d"] = float(hits.mean()) if len(hits) else None
        out[f"avg_return_{h}d"] = float(rets.mean()) if len(rets) else None
    return out


def _close_at_or_before(ts_array, close_array, ts: int) -> float | None:
    """
    Binary-search for the most recent close at or before ts.
    Returns None if no such bar exists or its close is NaN.
    """
    import numpy as np

    if len(ts_array) == 0:
        return None
    idx = np.searchsorted(ts_array, ts, side="right") - 1
    if idx < 0:
        return None
    val = close_array[idx]
    if pd.isna(val):
        return None
    return float(val)


def _empty() -> pd.DataFrame:
    return pd.DataFrame(columns=_OUT_COLS)
