"""Offline unit tests for the SQL cleaning + safety gate (no LLM, no database writes)."""

import pytest

from medibot.sql_rag import SQLSafetyError, clean_sql


def test_strips_markdown_fence_and_adds_limit():
    assert clean_sql("```sql\nSELECT COUNT(*) FROM claims;\n```") == "SELECT COUNT(*) FROM claims LIMIT 50"


def test_strips_sqlquery_prefix():
    assert clean_sql("SQLQuery: select status, count(*) from claims group by 1").startswith("select status")


def test_keeps_existing_limit():
    assert clean_sql("SELECT * FROM claims LIMIT 5").endswith("LIMIT 5")


def test_allows_cte():
    assert clean_sql("WITH x AS (SELECT * FROM claims) SELECT COUNT(*) FROM x").startswith("WITH x")


@pytest.mark.parametrize("bad", ["DROP TABLE claims", "DELETE FROM claims", "SELECT 1; SELECT 2",
                                 "UPDATE claims SET status='x'", "PRAGMA table_info(claims)", "explain SELECT 1"])
def test_rejects_unsafe(bad):
    with pytest.raises(SQLSafetyError):
        clean_sql(bad)
