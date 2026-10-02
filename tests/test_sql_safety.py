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
                                 "UPDATE claims SET status='x'", "PRAGMA table_info(claims)", "SELECT 1; DROP TABLE claims"])
def test_rejects_unsafe(bad):
    with pytest.raises(SQLSafetyError):
        clean_sql(bad)


def test_prose_before_statement_is_skipped():
    assert clean_sql("Here is the SQL:\nSELECT COUNT(*) FROM claims").startswith("SELECT COUNT(*)")


def test_prose_after_blank_line_is_dropped():
    sql = clean_sql("SELECT * FROM claims\n\nThis query counts rows.")
    assert sql == "SELECT * FROM claims LIMIT 50"


def test_replace_function_is_allowed_but_replace_into_is_not():
    assert "REPLACE(" in clean_sql("SELECT REPLACE(insurer, ' ', '_') FROM claims")
    with pytest.raises(SQLSafetyError):
        clean_sql("REPLACE INTO claims VALUES (1)")


def test_semicolon_inside_string_literal_is_not_a_split():
    assert clean_sql("SELECT * FROM claims WHERE insurer = 'A;B'").startswith("SELECT * FROM claims WHERE insurer = 'A;B'")


def test_only_the_select_statement_survives_leading_junk():
    """Anything before the first SELECT is treated as prose and dropped; only the extracted statement ever runs."""
    assert clean_sql("DROP TABLE claims; SELECT 1") == "SELECT 1 LIMIT 50"
