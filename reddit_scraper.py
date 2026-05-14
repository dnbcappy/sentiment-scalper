"""
Sentiment Scalper - Reddit ingestion
-------------------------------------
Pulls recent posts from a curated set of finance subreddits via PRAW
(the Python Reddit API Wrapper). Returns dicts in the same shape as
fetch_newsapi() so the rest of the ingest pipeline doesn't care which
source the data came from.

Setup (one-time):
  1. Sign in at https://reddit.com/prefs/apps
  2. "Create app" -> type: script -> redirect URI: http://localhost
  3. Save client_id (the short string under the app name) and client_secret
  4. Put them in .env as REDDIT_CLIENT_ID + REDDIT_CLIENT_SECRET
  5. Optionally override REDDIT_USER_AGENT (Reddit asks for descriptive UA)

Run cadence: called from sentiment_scalper.ingest() alongside NewsAPI on
every scraper cron. Rate limit is generous (~100 req/min OAuth) so we
never come close to hitting it with 8 cron runs/day.

Graceful degradation: if credentials are missing the function logs a
warning and returns []. The scraper still runs NewsAPI as normal.
"""

from __future__ import annotations

import logging
import os
from datetime import datetime, timedelta, timezone

logger = logging.getLogger(__name__)

# Curated set of finance-focused subreddits. Mix of stocks + crypto,
# active enough to produce real signal but not so niche they're empty.
REDDIT_SUBREDDITS = [
    "wallstreetbets",
    "stocks",
    "investing",
    "cryptocurrency",
    "CryptoMarkets",
    "Bitcoin",
    "ethfinance",
]

# Max posts to pull per subreddit per run. PRAW's per-request cap is 100.
# 100 * 7 subreddits = up to 700 posts per cron run, vs ~9 NewsAPI articles
# per query -> Reddit becomes the dominant data source going forward.
POSTS_PER_SUBREDDIT = 100

# Only accept posts within this lookback window. Matches the NewsAPI
# constant so signals computed across both sources share the same horizon.
LOOKBACK_HOURS = 168


def fetch_reddit() -> list[dict]:
    """Pull recent posts from configured subreddits.

    Returns a list of dicts shaped like fetch_newsapi() output:
        {id, source, subreddit, text, score, created_utc}

    Returns [] when REDDIT_CLIENT_ID / REDDIT_CLIENT_SECRET are missing
    (so the scraper can run NewsAPI-only without configuration).
    """
    client_id = os.getenv("REDDIT_CLIENT_ID", "").strip()
    client_secret = os.getenv("REDDIT_CLIENT_SECRET", "").strip()
    user_agent = os.getenv(
        "REDDIT_USER_AGENT",
        "sentiment-scalper/1.0 by github.com/dnbcappy",
    )

    if not client_id or not client_secret:
        logger.warning("Reddit credentials missing — skipping reddit fetch")
        return []

    # praw is imported lazily so the module can be imported without praw
    # installed (e.g., in lightweight contexts). All callers run the
    # full requirements.txt so this just keeps import-time fast.
    import praw

    reddit = praw.Reddit(
        client_id=client_id,
        client_secret=client_secret,
        user_agent=user_agent,
    )
    reddit.read_only = True

    cutoff_ts = int((datetime.now(timezone.utc) - timedelta(hours=LOOKBACK_HOURS)).timestamp())
    items: list[dict] = []

    for sub_name in REDDIT_SUBREDDITS:
        accepted = 0
        try:
            subreddit = reddit.subreddit(sub_name)
            for post in subreddit.new(limit=POSTS_PER_SUBREDDIT):
                created = int(post.created_utc)
                if created < cutoff_ts:
                    # `.new()` returns newest first — once we cross the
                    # lookback boundary, everything after is also older.
                    break
                title = post.title or ""
                body = post.selftext or ""
                items.append(
                    {
                        "id": f"reddit_{post.id}",
                        "source": "reddit",
                        "subreddit": sub_name,
                        "text": f"{title}\n\n{body}".strip(),
                        "score": int(post.score),
                        "created_utc": created,
                    }
                )
                accepted += 1
            logger.info("Fetched %d posts from r/%s", accepted, sub_name)
        except Exception:
            # Don't pass the exception object — its repr could include the
            # OAuth credentials in the Authorization header on some PRAW errors.
            logger.error("Failed to fetch from r/%s", sub_name)

    logger.info(
        "Reddit fetch totals: %d posts across %d subreddits", len(items), len(REDDIT_SUBREDDITS)
    )
    return items
