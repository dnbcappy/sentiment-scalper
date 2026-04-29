"""Tests for sentiment_scalper.py — ticker matching and ID migration."""

from __future__ import annotations

import sqlite3

from sentiment_scalper import _migrate_ids_v2, find_tickers

# ---------- find_tickers ----------


def test_find_tickers_matches_full_name():
    assert find_tickers("Apple announces new Mac") == {"AAPL"}
    assert find_tickers("Tesla earnings beat") == {"TSLA"}
    assert find_tickers("Microsoft cloud growth") == {"MSFT"}


def test_find_tickers_matches_symbol():
    assert find_tickers("AAPL stock rallies") == {"AAPL"}
    assert find_tickers("BTC pumps") == {"BTC"}


def test_find_tickers_matches_multiple():
    result = find_tickers("Bitcoin and Ethereum surge in tandem")
    assert "BTC" in result
    assert "ETH" in result


def test_find_tickers_case_insensitive():
    assert find_tickers("bitcoin breaks 100k") == {"BTC"}
    assert find_tickers("BITCOIN breaks 100k") == {"BTC"}


def test_find_tickers_empty_input():
    assert find_tickers("") == set()
    assert find_tickers(None) == set()  # type: ignore[arg-type]


def test_find_tickers_no_match():
    assert find_tickers("just some random text about cooking") == set()


def test_find_tickers_word_boundary():
    """Should not match substrings like 'eth' inside 'ether'-unrelated words."""
    # 'ETH' as a standalone word should match
    assert "ETH" in find_tickers("ETH is up 5%")
    # 'ethernet' should NOT trigger ETH (word boundary)
    assert "ETH" not in find_tickers("ethernet cable replaced")


# ---------- _migrate_ids_v2 ----------


def _setup_old_format_row(db_path: str, row_id: str, model: str) -> None:
    conn = sqlite3.connect(db_path)
    conn.execute(
        "INSERT INTO mentions "
        "(id, ticker, source, subreddit, text, score, model, "
        " compound, pos, neg, neu, created_utc, fetched_at) "
        "VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)",
        (row_id, "BTC", "newsapi", "test", "text", 0, model, 0.5, 0, 0, 0, 0, 0),
    )
    conn.commit()
    conn.close()


def test_migrate_ids_appends_model_suffix(db_path):
    _setup_old_format_row(db_path, "newsapi_abc__BTC", "vader")
    n = _migrate_ids_v2()
    assert n == 1

    conn = sqlite3.connect(db_path)
    rows = conn.execute("SELECT id FROM mentions").fetchall()
    conn.close()
    assert rows[0][0] == "newsapi_abc__BTC__vader"


def test_migrate_ids_idempotent(db_path):
    """Running migration twice yields no changes the second time."""
    _setup_old_format_row(db_path, "newsapi_abc__BTC", "vader")
    first = _migrate_ids_v2()
    second = _migrate_ids_v2()
    assert first == 1
    assert second == 0


def test_migrate_ids_skips_already_migrated_rows(db_path):
    _setup_old_format_row(db_path, "already__BTC__vader", "vader")
    n = _migrate_ids_v2()
    assert n == 0


def test_migrate_ids_handles_mixed_state(db_path):
    """Some rows old, some new — only the old ones get migrated."""
    _setup_old_format_row(db_path, "old__BTC", "vader")
    _setup_old_format_row(db_path, "new__BTC__vader", "vader")
    n = _migrate_ids_v2()
    assert n == 1

    conn = sqlite3.connect(db_path)
    ids = sorted(r[0] for r in conn.execute("SELECT id FROM mentions"))
    conn.close()
    assert ids == ["new__BTC__vader", "old__BTC__vader"]
