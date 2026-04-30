# Phase 5 — Deployment proposal

**Status:** proposal, not yet executed. Read top to bottom, then pick a path. No code yet.

The goal: make the dashboard publicly viewable so other people can see what the engine is firing on. The scraper has to keep running in the background. The data has to persist across restarts.

---

## Step 0 — Rotate the NewsAPI key first (blocker)

The current key was exposed during development. **Do this before any deploy** — once the dashboard is public, the key sitting in your local `.env` would have already been exposed if it ever leaked, and rotation is harder once a real audience exists.

**What you do:**

1. Email **`support@newsapi.org`** from the address registered with your NewsAPI account, using this template:

   > Subject: Key rotation request
   >
   > Hi,
   >
   > My API key on the free tier was inadvertently exposed in development logs.
   > Could you please rotate it?
   >
   > Thanks

2. They'll either issue a new key directly or ask you to regenerate via the dashboard. Either way you end up with a new `NEWSAPI_KEY` value.
3. Update your local `.env` with the new value.
4. **Do not commit it.** The new key only exists in:
   - Your local `.env` (gitignored, so safe)
   - The hosting platform's secret store (whichever we pick below)

If you don't get a response in ~48h, follow up. Don't deploy with the old key still active.

---

## Step 1 — The two real choices

Three independent decisions: where the **app** runs, where the **scraper** runs, where the **DB** lives. They're independent because the app/scraper can talk to the DB over the network.

### Option A — Streamlit Community Cloud + Supabase + GitHub Actions ($0)

| Component | Service                 | Why                                                       |
|-----------|-------------------------|-----------------------------------------------------------|
| Dashboard | Streamlit Cloud (free)  | Zero config, deploys directly from your GitHub repo.      |
| Scraper   | GitHub Actions cron     | Free for public repos, native to where your code lives.   |
| DB        | Supabase Postgres (free)| 500MB free tier, plenty for years of mention data.        |

**Pros:** Free. Setup measured in minutes, not hours. Dashboard auto-redeploys on `git push`.
**Cons:** Streamlit Cloud apps sleep after ~10 min of inactivity (cold start ~5s). Public-only — anyone with the URL can view. 1GB RAM ceiling. You're touching three different vendors.

### Option B — Firebase Hosting + Cloud Run + Cloud SQL (~$5-15/month)

| Component | Service              | Why                                                                |
|-----------|----------------------|--------------------------------------------------------------------|
| Dashboard | Cloud Run (Streamlit)| Containerized, scales to zero when idle.                           |
| Scraper   | Cloud Run + Cloud Scheduler | Same container or a separate job, triggered every N minutes. |
| DB        | Cloud SQL Postgres   | Smallest instance ~$10/mo, lower if you use Firestore instead.     |

**Pros:** Single vendor (you already use Firebase for cappystudios.dev). Production-grade — IAM, logging, backups, custom domain ergonomics. No sleep behavior.
**Cons:** Costs real money once Cloud SQL is on (Cloud Run itself is mostly free at this traffic level). Setup is hours, not minutes — Dockerfile, service accounts, IAM bindings. Overkill if no one's actually using the app yet.

### Sidebar: why not "just keep SQLite"?

Cloud platforms with multiple containers don't share local disk. Streamlit Cloud's container also gets reset between deploys, so a SQLite file would lose all data. The DB has to be a network service (Postgres) once we leave your laptop. The schema migration is simple — change a few `INSERT OR REPLACE` to `INSERT ... ON CONFLICT ... DO UPDATE` and swap `sqlite3` calls for `psycopg` or SQLAlchemy.

---

## Recommendation: **Option A (Streamlit Cloud + Supabase + GitHub Actions)**

For where this project is right now:

1. **It's not earning money yet** — paying $10/mo for Cloud SQL is premature.
2. **You can ship in an evening** — I've estimated ~2 hours of work, mostly waiting for builds.
3. **Migration to Option B later is straightforward** — Cloud Run can pull the same code; only the DB connection string changes if we use Postgres on both sides.
4. **The cold-start tax is irrelevant** for a project people will check once or twice a day.

The only reason to skip A and go to B today: if you specifically want this on cappystudios.dev infrastructure for portfolio reasons. That's a real reason, just not a code one.

---

## Concrete steps if you pick Option A

These are the things I'd do, in order. We don't write any code until you greenlight the plan.

1. **Sign up for Supabase** (free tier, no card). Create a project, grab the connection string (`postgresql://...`).
2. **Migrate the schema** — write a small `migrations/0001_initial.sql` that creates `mentions` and `prices` tables with Postgres-compatible syntax. Run it once against Supabase.
3. **Refactor the DB layer** — currently each module opens `sqlite3.connect(DB_PATH)` directly. Swap for `psycopg` (or `psycopg2-binary`) and a connection-string env var. Keep SQLite as a fallback for local dev so tests still run on a temp file.
4. **One-shot data migration** — dump local `sentiment.db` rows to Supabase. Probably a 10-line script.
5. **Streamlit Cloud setup** — connect GitHub repo, set `DATABASE_URL` and `NEWSAPI_KEY` as Streamlit secrets via the UI. Deploy.
6. **GitHub Actions cron** — a `.github/workflows/scrape.yml` that runs `python sentiment_scalper.py` every hour, with `NEWSAPI_KEY` and `DATABASE_URL` as repo secrets.
7. **Smoke test the URL**, check logs, share the link.

Estimated effort: ~3-4 hours with debugging.

---

## What I'd NOT do during Phase 5

- Add authentication. Public read-only is fine for a demo.
- Add error tracking (Sentry, etc.) — premature.
- Set up a custom domain. The default `*.streamlit.app` URL is fine for v1.
- Move to FinBERT in cloud. Torch on Streamlit Cloud is borderline — model takes ~5s to load and eats into the 1GB RAM. Stick with VADER in cloud, run FinBERT locally if you want to compare.

---

## What you decide right now

1. **Option A or Option B?** Pick one. If A, no further input needed — I'll execute steps 1-7 above.
2. **Once the new NewsAPI key is in hand**, do you want me to also rotate the variable name to `NEWS_API_KEY` (cleaner) or leave it as `NEWSAPI_KEY` (no churn)? Tiny thing, just flagging.
3. **Anything you want excluded** from the public dashboard? Right now everything is visible to anyone with the URL.

Say the word and I'll start on the migration.
