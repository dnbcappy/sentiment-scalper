"""
Sentiment Scalper - Dashboard
-----------------------------
Streamlit dashboard for visualizing the SQLite database produced
by sentiment_scalper.py.

Setup:
    pip install -r requirements.txt

Run:
    streamlit run dashboard.py
"""

import os
import sqlite3
from datetime import datetime, timedelta, timezone

import altair as alt
import pandas as pd
import streamlit as st

from backtest import compute_hit_rates
from prices import get_prices, update_prices
from signals import compute_signals, list_engines

DB_PATH = os.getenv(
    "DB_PATH",
    os.path.join(os.path.dirname(os.path.abspath(__file__)), "sentiment.db"),
)

st.set_page_config(page_title="Sentiment Scalper", layout="wide")

# ---------- Data loading ----------


@st.cache_data(ttl=60)
def load_data(hours: int, model: str | None = None) -> pd.DataFrame:
    cutoff = int((datetime.now(timezone.utc) - timedelta(hours=hours)).timestamp())
    sql = "SELECT * FROM mentions WHERE created_utc >= ?"
    params: list = [cutoff]
    if model is not None:
        sql += " AND model = ?"
        params.append(model)
    sql += " ORDER BY created_utc DESC"

    conn = sqlite3.connect(DB_PATH)
    try:
        df = pd.read_sql_query(sql, conn, params=params)
    finally:
        conn.close()
    if not df.empty:
        df["timestamp"] = pd.to_datetime(df["created_utc"], unit="s", utc=True)
    return df


@st.cache_data(ttl=60)
def load_signals(db_path: str, model: str | None = None) -> pd.DataFrame:
    return compute_signals(db_path, model=model)


@st.cache_data(ttl=3600)
def refresh_prices(db_path: str) -> int:
    return update_prices(db_path)


@st.cache_data(ttl=3600)
def load_prices(db_path: str) -> pd.DataFrame:
    return get_prices(db_path)


@st.cache_data(ttl=600)
def load_hit_rates(db_path: str, threshold: float, model: str | None = None) -> pd.DataFrame:
    return compute_hit_rates(db_path, threshold, model=model)


# ---------- Price refresh (cached 1h) ----------

try:
    with st.spinner("Refreshing prices (cached 1h)..."):
        refresh_prices(DB_PATH)
except Exception as e:
    st.warning(f"Price refresh failed: {e}. Continuing with cached data.")

# ---------- Sidebar ----------

st.sidebar.title("Filters")
hours = st.sidebar.slider("Lookback (hours)", min_value=1, max_value=168, value=24)
signal_threshold = st.sidebar.slider(
    "Signal threshold", min_value=0.0, max_value=5.0, value=1.5, step=0.1
)

available_engines = list_engines(DB_PATH)
if available_engines:
    selected_engine = st.sidebar.selectbox("Sentiment engine", available_engines, index=0)
else:
    selected_engine = None

df = load_data(hours, model=selected_engine)

if df.empty:
    st.title("📈 Sentiment Scalper")
    st.warning("No data yet. Run `python sentiment_scalper.py` first.")
    st.stop()

tickers = sorted(df["ticker"].unique())
selected = st.sidebar.multiselect("Tickers", tickers, default=tickers)
sources = sorted(df["source"].unique())
selected_sources = st.sidebar.multiselect("Sources", sources, default=sources)

df = df[df["ticker"].isin(selected) & df["source"].isin(selected_sources)]

if df.empty:
    st.warning("No data matches the current filters.")
    st.stop()

# ---------- Header ----------

st.title("📈 Sentiment Scalper")
st.caption(f"Last {hours}h • {len(df):,} mentions • engine: {selected_engine or 'n/a'}")

# ---------- Active signals ----------

st.subheader("Active signals right now")

signals_df = load_signals(DB_PATH, model=selected_engine)
active = (
    signals_df[signals_df["signal"].abs() >= signal_threshold]
    if not signals_df.empty
    else signals_df
)

if active.empty:
    st.info("No signals above threshold. Lower the slider or run the scraper to collect more data.")
else:
    st.dataframe(
        active.style.format(
            {
                "signal": "{:+.2f}",
                "volume_z": "{:+.2f}",
                "sentiment_z": "{:+.2f}",
                "current_sent": "{:+.3f}",
            }
        ),
        use_container_width=True,
        hide_index=True,
    )

# ---------- KPIs ----------

c1, c2, c3, c4 = st.columns(4)
c1.metric("Mentions", f"{len(df):,}")
c2.metric("Avg sentiment", f"{df['compound'].mean():+.3f}")
c3.metric("Bullish (>0.2)", int((df["compound"] > 0.2).sum()))
c4.metric("Bearish (<-0.2)", int((df["compound"] < -0.2).sum()))

# ---------- Per-ticker summary ----------

st.subheader("Per-ticker summary")
summary = (
    df.groupby("ticker")
    .agg(
        mentions=("id", "count"),
        avg_sentiment=("compound", "mean"),
        bullish=("compound", lambda s: int((s > 0.2).sum())),
        bearish=("compound", lambda s: int((s < -0.2).sum())),
    )
    .sort_values("mentions", ascending=False)
)
summary["bull_bear_ratio"] = (summary["bullish"] + 1) / (summary["bearish"] + 1)
st.dataframe(
    summary.style.format(
        {
            "avg_sentiment": "{:+.3f}",
            "bull_bear_ratio": "{:.2f}",
        }
    ),
    use_container_width=True,
)

# ---------- Sentiment over time ----------

st.subheader("Avg sentiment per ticker (1h buckets)")
sent_ts = (
    df.set_index("timestamp").groupby("ticker")["compound"].resample("1h").mean().unstack("ticker")
)
st.line_chart(sent_ts)

# ---------- Mention volume ----------

st.subheader("Mention volume per ticker (1h buckets)")
vol_ts = (
    df.set_index("timestamp")
    .groupby("ticker")["id"]
    .resample("1h")
    .count()
    .unstack("ticker")
    .fillna(0)
)
st.bar_chart(vol_ts)

# ---------- Sentiment vs Price (per ticker) ----------

st.subheader("Sentiment vs Price")
prices_df = load_prices(DB_PATH)

if prices_df.empty:
    st.info("No price data cached yet.")
else:
    overlay_ticker = st.selectbox(
        "Ticker",
        options=sorted(set(df["ticker"].unique()) & set(prices_df["ticker"].unique())),
        key="overlay_ticker",
    )

    # Sentiment: mentions for this ticker, hourly mean within the lookback window
    sent_one = (
        df[df["ticker"] == overlay_ticker]
        .set_index("timestamp")["compound"]
        .resample("1h")
        .mean()
        .dropna()
        .reset_index()
    )

    # Prices: same ticker, restrict to the lookback window for visual coherence
    since_ts = int((datetime.now(timezone.utc) - timedelta(hours=hours)).timestamp())
    price_one = prices_df[(prices_df["ticker"] == overlay_ticker) & (prices_df["ts"] >= since_ts)][
        ["timestamp", "close"]
    ].dropna()

    if sent_one.empty or price_one.empty:
        st.info(f"Not enough overlapping data for {overlay_ticker} in the last {hours}h.")
    else:
        sent_layer = (
            alt.Chart(sent_one)
            .mark_line(color="#4c78a8")
            .encode(
                x=alt.X("timestamp:T", title="Time"),
                y=alt.Y(
                    "compound:Q",
                    title="Avg sentiment",
                    axis=alt.Axis(titleColor="#4c78a8"),
                    scale=alt.Scale(zero=False),
                ),
            )
        )
        price_layer = (
            alt.Chart(price_one)
            .mark_line(color="#f58518", point=True)
            .encode(
                x=alt.X("timestamp:T"),
                y=alt.Y(
                    "close:Q",
                    title="Close price (USD)",
                    axis=alt.Axis(titleColor="#f58518"),
                    scale=alt.Scale(zero=False),
                ),
            )
        )
        chart = (
            alt.layer(sent_layer, price_layer)
            .resolve_scale(y="independent")
            .properties(
                height=400, title=f"{overlay_ticker}: sentiment (1h) vs price (daily close)"
            )
        )
        st.altair_chart(chart, use_container_width=True)
        st.caption(
            "Note: sentiment is hourly, price is daily close. A 24h lookback may show only 1 price point."
        )

# ---------- Did it work? — backtest hit rate ----------

st.subheader("Did it work? — backtest hit rate")
st.caption(
    f"For each past signal at |signal| ≥ {signal_threshold:.1f}, did the price move "
    "in the predicted direction over the next 1d / 3d / 7d? "
    "Horizons that haven't elapsed yet are excluded (still-open positions)."
)

hit_df = load_hit_rates(DB_PATH, signal_threshold, model=selected_engine)
if hit_df.empty:
    st.info("No backtest results yet — need historical signals plus cached price data.")
else:
    st.dataframe(
        hit_df.style.format(
            {
                "hit_rate_1d": "{:.1%}",
                "hit_rate_3d": "{:.1%}",
                "hit_rate_7d": "{:.1%}",
                "avg_return_1d": "{:+.2%}",
                "avg_return_3d": "{:+.2%}",
                "avg_return_7d": "{:+.2%}",
            },
            na_rep="—",
        ),
        use_container_width=True,
        hide_index=True,
    )

# ---------- Engine comparison (only when 2+ engines have data) ----------

if len(available_engines) >= 2:
    st.subheader("Engine comparison")
    st.caption(
        "Same backtest, run separately per sentiment engine. Use this to see whether "
        "one engine's signals beat the other on the same articles."
    )
    compare_frames = []
    for engine_name in available_engines:
        per_engine = load_hit_rates(DB_PATH, signal_threshold, model=engine_name)
        if per_engine.empty:
            continue
        per_engine = per_engine.copy()
        per_engine.insert(0, "engine", engine_name)
        compare_frames.append(per_engine)

    if not compare_frames:
        st.info("No backtest results in either engine yet.")
    else:
        compare_df = pd.concat(compare_frames, ignore_index=True)
        st.dataframe(
            compare_df.style.format(
                {
                    "hit_rate_1d": "{:.1%}",
                    "hit_rate_3d": "{:.1%}",
                    "hit_rate_7d": "{:.1%}",
                    "avg_return_1d": "{:+.2%}",
                    "avg_return_3d": "{:+.2%}",
                    "avg_return_7d": "{:+.2%}",
                },
                na_rep="—",
            ),
            use_container_width=True,
            hide_index=True,
        )

# ---------- Recent mentions ----------

st.subheader("Recent mentions")
recent = df[["timestamp", "ticker", "source", "subreddit", "compound", "text"]].head(100).copy()
recent["text"] = recent["text"].fillna("").apply(lambda t: t[:240] + "…" if len(t) > 240 else t)
st.dataframe(recent, use_container_width=True, hide_index=True)
