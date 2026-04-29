"""
Sentiment Scalper - Database engine factory
--------------------------------------------
Centralizes database connection logic so the same code works against either
local SQLite (default for development) or hosted Postgres (for production).

Selection: the DATABASE_URL environment variable.
  - Unset / empty       -> local SQLite at ./sentiment.db (next to this file)
  - sqlite:///<path>    -> SQLite at the given absolute path
  - postgresql://...    -> hosted Postgres (Supabase, Cloud SQL, etc.)
  - postgres://...      -> normalized to postgresql:// (some hosts emit this)

Same SQL works in both backends. SQLite 3.24+ (released 2018, included in
all current Python builds) supports the INSERT ... ON CONFLICT syntax that
Postgres uses, so portable upserts are clean to write.

Usage:
    from sqlalchemy import text
    from db import get_engine

    with get_engine().connect() as conn:
        df = pd.read_sql_query(text("..."), conn, params={...})

    with get_engine().begin() as conn:   # begin() auto-commits at exit
        conn.execute(text("INSERT ..."), {...})
"""

from __future__ import annotations

import logging
import os
from functools import lru_cache

from sqlalchemy import Engine, create_engine

logger = logging.getLogger(__name__)

_DEFAULT_SQLITE_PATH = os.path.join(os.path.dirname(os.path.abspath(__file__)), "sentiment.db")


def get_database_url() -> str:
    """Resolve the configured DATABASE_URL, falling back to local SQLite."""
    url = os.getenv("DATABASE_URL", "").strip()
    if not url:
        return f"sqlite:///{_DEFAULT_SQLITE_PATH}"
    # Some hosts (Heroku-style) emit postgres://; SQLAlchemy 2.0 only
    # accepts postgresql://.
    if url.startswith("postgres://"):
        url = "postgresql://" + url[len("postgres://") :]
    return url


@lru_cache(maxsize=1)
def get_engine() -> Engine:
    """Process-wide SQLAlchemy engine for the configured database.

    Cached: a single engine per process is the right pattern. Pool config
    is light — pool_pre_ping handles dropped connections on free Postgres
    tiers that idle out.
    """
    url = get_database_url()
    logger.debug("Database URL dialect: %s", url.split("://", 1)[0])
    return create_engine(url, pool_pre_ping=True, future=True)


def is_postgres() -> bool:
    return get_database_url().startswith("postgresql")


def reset_engine() -> None:
    """Clear the cached engine. Tests use this between fixtures."""
    get_engine.cache_clear()
