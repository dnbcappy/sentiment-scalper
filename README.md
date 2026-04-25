# Sentiment Scalper

Tracks news sentiment for selected stocks and cryptocurrencies, scores it with VADER (or FinBERT, optional), stores everything in SQLite, and visualizes the result with a Streamlit dashboard.

Currently tracking: BTC, ETH, USDT, USDC, SPY, AAPL, TSLA, NVDA, MSFT.

## Stack

- **Python 3.10+** — main language
- **NewsAPI** (free tier, 1,000 req/day) — news source covering Cointelegraph, CoinDesk, Decrypt, Reuters, Bloomberg, CNBC and similar publishers
- **VADER** — fast rule-based sentiment scoring (default)
- **FinBERT** — finance-tuned transformer scoring (optional, ~440MB download)
- **SQLite** — local storage, no setup required
- **Streamlit** — interactive dashboard
- **python-dotenv** — environment variable loading

## Setup

```bash
# 1. Clone
git clone https://github.com/YOUR_USERNAME/sentiment-scalper.git
cd sentiment-scalper

# 2. Create and activate a virtual environment
python -m venv venv
# Windows (Git Bash): source venv/Scripts/activate
# macOS / Linux:      source venv/bin/activate

# 3. Install dependencies
pip install -r requirements.txt

# 4. Configure
cp .env.example .env
# then edit .env and add your NewsAPI key
```

Get a free NewsAPI key at https://newsapi.org/register.

## Usage

Ingest fresh data:

```bash
python sentiment_scalper.py
```

Open the dashboard:

```bash
streamlit run dashboard.py
```

Visit `http://localhost:8501` in your browser. Use the sidebar slider to control the lookback window (1–168 hours).

## How it works

1. `sentiment_scalper.py` queries NewsAPI once per ticker, deduplicates by URL, filters to articles that actually mention a tracked ticker, scores each with the chosen sentiment engine, and inserts results into `sentiment.db`.
2. `dashboard.py` reads from the same SQLite file and renders KPIs, a per-ticker summary table, sentiment over time, mention volume, and the most recent mentions.

## Project layout

```
sentiment-scalper/
├── sentiment_scalper.py   # ingest + score pipeline
├── dashboard.py           # Streamlit dashboard
├── requirements.txt       # Python dependencies
├── .env.example           # template for credentials (commit this)
├── .env                   # real credentials (gitignored, never commit)
├── .gitignore
└── sentiment.db           # local SQLite DB (gitignored, generated at runtime)
```

## Roadmap

- [ ] Signal scoring (volume × directional sentiment vs rolling baseline)
- [ ] Add price overlay via `yfinance` for backtesting
- [ ] Swap default engine to FinBERT once benchmarked against VADER
- [ ] Publisher blocklist for noise sources (PyPI, Slashdot, etc.)
- [ ] RSS fetcher as a second free data source
- [ ] Scheduled runs (Windows Task Scheduler / cron)

## Notes

- NewsAPI free tier limits articles to roughly the last 30 days. The scalper queries the last 7 days by default — adjust the `hours=168` in `fetch_newsapi` if you want a different window.
- Free tier is rate-limited to 1,000 requests/day across all queries. With 9 tickers per run, that caps you at roughly 100 runs/day — way more than needed for a 30-minute polling cycle.

## License

Personal project. Do whatever you want with it.
