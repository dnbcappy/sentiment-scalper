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

import sqlite3
from datetime import datetime, timedelta, timezone

import pandas as pd
import streamlit as st

DB_PATH = "sentiment.db"

st.set_page_config(page_title="Sentiment Scalper", layout="wide")

# ---------- Data loading ----------

@st.cache_data(ttl=60)
def load_data(hours: int) -> pd.DataFrame:
    cutoff = int((datetime.now(timezone.utc) - timedelta(hours=hours)).timestamp())
    conn = sqlite3.connect(DB_PATH)
    try:
        df = pd.read_sql_query(
            "SELECT * FROM mentions WHERE created_utc >= ? ORDER BY created_utc DESC",
            conn, params=(cutoff,),
        )
    finally:
        conn.close()
    if not df.empty:
        df["timestamp"] = pd.to_datetime(df["created_utc"], unit="s", utc=True)
    return df

# ---------- Sidebar ----------

st.sidebar.title("Filters")
hours = st.sidebar.slider("Lookback (hours)", min_value=1, max_value=168, value=24)
df = load_data(hours)

if df.empty:
    st.title("📈 Sentiment Scalper")
    st.warning("No data yet. Run `python sentiment_scalper.py` first.")
    st.stop()

tickers = sorted(df["ticker"].unique())
selected = st.sidebar.multiselect("Tickers", tickers, default=tickers)
sources  = sorted(df["source"].unique())
selected_sources = st.sidebar.multiselect("Sources", sources, default=sources)

df = df[df["ticker"].isin(selected) & df["source"].isin(selected_sources)]

# ---------- Header / KPIs ----------

st.title("📈 Sentiment Scalper")
st.caption(f"Last {hours}h • {len(df):,} mentions • models: {', '.join(df['model'].dropna().unique()) or 'n/a'}")

c1, c2, c3, c4 = st.columns(4)
c1.metric("Mentions", f"{len(df):,}")
c2.metric("Avg sentiment", f"{df['compound'].mean():+.3f}")
c3.metric("Bullish (>0.2)", int((df["compound"] >  0.2).sum()))
c4.metric("Bearish (<-0.2)", int((df["compound"] < -0.2).sum()))

# ---------- Per-ticker summary ----------

st.subheader("Per-ticker summary")
summary = (
    df.groupby("ticker")
      .agg(mentions=("id", "count"),
           avg_sentiment=("compound", "mean"),
           bullish=("compound", lambda s: int((s >  0.2).sum())),
           bearish=("compound", lambda s: int((s < -0.2).sum())))
      .sort_values("mentions", ascending=False)
)
summary["bull_bear_ratio"] = (summary["bullish"] + 1) / (summary["bearish"] + 1)
st.dataframe(summary.style.format({
    "avg_sentiment": "{:+.3f}",
    "bull_bear_ratio": "{:.2f}",
}), use_container_width=True)

# ---------- Sentiment over time ----------

st.subheader("Avg sentiment per ticker (1h buckets)")
sent_ts = (
    df.set_index("timestamp")
      .groupby("ticker")["compound"]
      .resample("1h")
      .mean()
      .unstack("ticker")
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

# ---------- Recent mentions ----------

st.subheader("Recent mentions")
recent = df[["timestamp", "ticker", "source", "subreddit", "compound", "text"]].head(100).copy()
recent["text"] = recent["text"].str.slice(0, 240) + "…"
st.dataframe(recent, use_container_width=True, hide_index=True)
