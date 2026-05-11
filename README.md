# Sentiment Scalper

Real-time news sentiment signal engine for stocks & crypto with backtesting and live dashboard. It pulls headlines from NewsAPI, scores them with VADER (or FinBERT), buckets them into a volume-weighted z-score signal, and validates the result against historical price moves. A Streamlit dashboard surfaces what's firing right now and how past signals actually performed.

Currently tracking: **BTC, ETH, USDT, USDC, SPY, AAPL, TSLA, NVDA, MSFT.**

## 🔗 Live Demo
https://sentiment-scalper.streamlit.app/

<img width="100%" alt="Dashboard" src="https://github.com/user-attachments/assets/884beebd-131b-4bd3-aa61-4d3c0357904a" />

The deployed instance runs **autonomously on $0/month** — Supabase Postgres for storage, GitHub Actions cron for ingestion every 3 hours, Streamlit Community Cloud for the public dashboard.

---

## Stack

- **Python 3.10+**
- **NewsAPI** Developer free tier (~100 req/day, 7-day article window)
- **VADER** rule-based sentiment (default) or **FinBERT** finance-tuned transformer (optional, ~440MB model + transformers/torch)
- **yfinance** for daily OHLC
- **SQLAlchemy 2.0** over **SQLite** (local) or **Postgres** (deployed) — same code, swapped via `DATABASE_URL`
- **Streamlit + Altair + Plotly** dashboard
- **GitHub Actions** for the scheduled scraper
- **pytest + ruff** for tests and linting

---

## Quickstart

```bash
# 1. Clone and enter
git clone https://github.com/dnbcappy/sentiment-scalper.git
cd sentiment-scalper

# 2. Create and activate a virtual environment
python -m venv venv
source venv/Scripts/activate    # Windows (Git Bash)
# source venv/bin/activate       # macOS / Linux

# 3. Install runtime dependencies
pip install -r requirements.txt

# 4. Configure
cp .env.example .env
# edit .env and add your NewsAPI key — get one free at https://newsapi.org/register

# 5. Ingest some data
python sentiment_scalper.py

# 6. Open the dashboard
streamlit run dashboard.py
# visit http://localhost:8501
```

---

## What the dashboard shows

<p align="center">
  <img src="https://github.com/user-attachments/assets/09bf22e4-efe6-4ca3-920b-e7b4b25ba804" width="95%" />
</p>

- **News Sentiment Index** - a 0-100 gauge (Bull/Bear bands) computed from average sentiment + bullish/bearish article pressure over the last 48h. Note: this measures news media tone, not market behavior, so it differs from the price-based CMC / Alternative.me Crypto Fear & Greed Index.
- **Active signals right now** - tickers whose current 6-hour mention activity deviates from their 7-day baseline by more than the threshold.
- **KPIs** - total mentions, avg sentiment, bullish / bearish counts in the lookback window.
- **Per-ticker summary** - mention count, avg sentiment, bull/bear ratio.
- **Sentiment over time / Mention volume** - 1-hour bucketed time series.
- **Sentiment vs Price** - per-ticker dual-axis chart of hourly sentiment overlayed on daily close.
- **Did it work? — backtest hit rate** - for each historical signal, did the price move in the predicted direction over 1d / 3d / 7d?
- **Engine comparison** - automatically appears when both VADER and FinBERT have scored articles, side-by-side hit rates.

<p align="center">
  <img src="https://github.com/user-attachments/assets/89b1d782-9f59-4fd4-8a88-4606997cca30" width="95%" />
</p>
---

## How the signal works

For each ticker, the engine bins mentions into non-overlapping 6-hour buckets. The previous 7 days form the baseline distribution (mean and standard deviation) for both mention count and average sentiment.

```
volume_z    = (current_count    - baseline_count_mean) / baseline_count_std
sentiment_z = (current_avg_sent - baseline_sent_mean ) / baseline_sent_std

signal = sign(sentiment_z) * |volume_z|   when |volume_z| > 1.5
signal = 0                                 otherwise
```

The volume gate is the important part. Sentiment on normal volume is usually noise — coverage of a hot ticker is what predicts moves, and the direction of that coverage tells you which way. Tickers with fewer than 3 historical buckets are excluded (not enough data for a stable z-score).

The backtest walks every completed past bucket, computes the signal as if it were live at that moment, and compares the entry price to closes at +1d, +3d, +7d. Horizons that haven't elapsed yet are reported as "still open" and excluded from hit rates rather than counted as misses.

---

## Engine comparison

Both engines can score the same articles, with one row per (article, ticker, model) combination. To enable FinBERT:

```bash
pip install transformers torch     # ~600MB-2GB depending on platform
SENTIMENT_MODEL=finbert python sentiment_scalper.py
```

After the first FinBERT run, the dashboard's **Engine comparison** section appears automatically. Hit rates are computed per engine on the same backtest universe.

> Heads up: meaningful comparison needs **30+ historical signals per engine**. With a fresh DB and the free NewsAPI tier, expect 2-3 weeks of regular scraping before the comparison is statistically informative.

---

## Project layout

```
sentiment-scalper/
├── sentiment_scalper.py       # ingest + score pipeline (CLI entry)
├── dashboard.py               # Streamlit dashboard
├── signals.py                 # z-score signal engine + history walker
├── prices.py                  # yfinance fetcher + cache
├── backtest.py                # historical signal -> hit rate
├── index.py                   # News Sentiment Index gauge engine
├── db.py                      # SQLAlchemy engine factory (SQLite / Postgres)
├── tests/                     # pytest suite
│   ├── conftest.py
│   ├── test_signals.py
│   ├── test_backtest.py
│   ├── test_prices.py
│   ├── test_index.py
│   └── test_sentiment_scalper.py
├── migrations/0001_initial.sql       # Postgres schema for hosted deploys
├── scripts/migrate_local_to_remote.py # one-shot SQLite -> Postgres dump
├── .github/workflows/scrape.yml      # hourly cron (every 3h)
├── docs/DEPLOYMENT.md                # hosted-deploy walkthrough
├── requirements.txt           # runtime deps
├── requirements-dev.txt       # ruff, pytest
├── pyproject.toml             # ruff + pytest config
├── LICENSE                    # MIT
├── .env.example               # credentials template (commit this)
├── .env                       # real credentials (gitignored)
├── .gitignore
└── sentiment.db               # local SQLite DB (gitignored, generated)
```

---

## Development

```bash
# Install dev tools
pip install -r requirements-dev.txt

# Run tests
pytest

# Lint and format
ruff check .
ruff format .
```

Tests cover the signal engine, backtest math, ID migration, ticker matching, and the price cache layer. yfinance and Streamlit's render layer are not unit-tested — they're verified end-to-end via Streamlit's `AppTest` framework when needed.

---

## Configuration

All configuration is via environment variables in `.env`:

| Variable          | Default       | Notes                                                              |
|-------------------|---------------|--------------------------------------------------------------------|
| `NEWSAPI_KEY`     | (required)    | Free key at https://newsapi.org/register.                          |
| `SENTIMENT_MODEL` | `vader`       | `vader` or `finbert`. FinBERT requires `transformers` + `torch`.   |
| `DATABASE_URL`    | `sqlite:///./sentiment.db`       | Set to `postgresql://...` to use a hosted Postgres instead. |

---

## Operational notes

- **Free NewsAPI tier**: 100 requests/day on the Developer plan, articles limited to the last 7 days. Each scraper run does 9 requests (one per ticker), so the cron is set to every 3 hours = 8 runs × 9 = 72 req/day with headroom for manual runs.
- **Idempotent ingestion**: `INSERT ... ON CONFLICT` (works on both SQLite and Postgres) means re-running the scraper is safe — duplicates are silently dropped and today's price bar updates in place.
- **Article IDs** include the model suffix (`newsapi_<hash>__<ticker>__<model>`) so the same article can carry independent VADER and FinBERT scores. Existing rows are migrated in-place on the next `init_db()`.
- **Backtest accuracy**: entry/exit prices use "last close at or before" — simple and realistic. Weekend signals on stocks may show 0% return at short horizons because Friday's close is the only data; crypto trades 24/7 so this only affects equities. With more data this washes out in the aggregate.

---

## Roadmap

- [x] Phase 1: audit and clean (deterministic IDs, magic-number constants, dashboard fixes)
- [x] Phase 2: volume-weighted z-score signal engine
- [x] Phase 3: yfinance price overlay + 1d/3d/7d backtest
- [x] Phase 4: model-aware schema + engine comparison panel
- [x] Phase 5: deployment (Streamlit Cloud + Supabase Postgres + GitHub Actions cron)
- [x] Publisher whitelist for noise reduction (financial sources only)
- [x] News Sentiment Index gauge
- [ ] FinBERT comparison once enough data has accumulated (~3-4 weeks)
- [ ] Custom domain (web dev / SaaS brand — TBD)

---

## License

This project is licensed under the MIT License.

## Disclaimer
This project is for educational and research purposes only. 
Not financial advice. Data sourced from third-party APIs (NewsAPI, Yahoo Finance).
