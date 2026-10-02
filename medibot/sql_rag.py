"""SQL RAG: natural-language question -> SQL (LLM) -> clean -> execute read-only -> phrase the result (LLM).

`sql_rag_chain(question) -> str` is the plain function the assignment asks for; `run_sql_rag`
returns the same plus the SQL and rows so the API and README can show the work.

Safety: the SQLite connection is opened read-only, only ONE SELECT/WITH statement is accepted,
data-changing keywords are rejected, and a LIMIT is added when missing. The schema text is
built from PRAGMA at startup (tables, columns, allowed categorical values, date ranges) so the
model uses the exact values that exist ('escalated', not 'Escalated').
"""

from __future__ import annotations

import re
import sqlite3
from dataclasses import dataclass, field
from functools import lru_cache

from medibot.config import DB_PATH
from medibot.rag.llm import chat

FORBIDDEN = re.compile(r"\b(insert|update|delete|drop|alter|create|replace|attach|detach|pragma|vacuum|truncate)\b", re.I)
MAX_ROWS = 50


class SQLSafetyError(ValueError):
    pass


@dataclass
class SQLResult:
    answer: str
    sql: str
    columns: list[str] = field(default_factory=list)
    rows: list[tuple] = field(default_factory=list)


def connect() -> sqlite3.Connection:
    return sqlite3.connect(f"file:{DB_PATH}?mode=ro", uri=True, timeout=5)


@lru_cache(maxsize=1)
def schema_summary() -> str:
    """Tables, columns, low-cardinality values and date ranges, read from the database itself."""
    con = connect()
    lines: list[str] = []
    for (table,) in con.execute("SELECT name FROM sqlite_master WHERE type='table' ORDER BY name"):
        n = con.execute(f"SELECT COUNT(*) FROM {table}").fetchone()[0]
        lines.append(f"TABLE {table} ({n} rows)")
        for _, col, ctype, *_ in con.execute(f"PRAGMA table_info({table})"):
            distinct = con.execute(f"SELECT COUNT(DISTINCT {col}) FROM {table}").fetchone()[0]
            if distinct <= 12:
                vals = [r[0] for r in con.execute(f"SELECT DISTINCT {col} FROM {table} WHERE {col} IS NOT NULL ORDER BY 1")]
                lines.append(f"  {col} {ctype}: one of {vals}")
            elif col.endswith("_date"):
                lo, hi = con.execute(f"SELECT MIN({col}), MAX({col}) FROM {table}").fetchone()
                nulls = con.execute(f"SELECT COUNT(*) FROM {table} WHERE {col} IS NULL").fetchone()[0]
                note = f", NULL in {nulls} rows" if nulls else ""
                lines.append(f"  {col} {ctype}: ISO date text, {lo} .. {hi}{note}")
            else:
                lines.append(f"  {col} {ctype}")
    con.close()
    return "\n".join(lines)


SQL_SYSTEM = """You translate questions into SQLite SQL for MediAssist's operations database.
Output exactly ONE SELECT (or WITH ... SELECT) statement and nothing else: no explanation, no markdown.
Rules:
- Use only the tables and columns in the schema; use categorical values exactly as listed.
- Dates are ISO text 'YYYY-MM-DD'. The data covers 2024, so interpret relative phrases
  ("last month", "this year") against the latest date in the relevant table using
  strftime and a MAX(...) subquery, never against today's date.
- Amounts are in INR. Prefer GROUP BY + ORDER BY for "which ... most" questions.
- Never modify data."""

FEW_SHOTS = [
    (
        "How many billing claims were escalated last month?",
        "SELECT COUNT(*) AS escalated_claims FROM claims WHERE status = 'escalated' "
        "AND strftime('%Y-%m', submitted_date) = (SELECT strftime('%Y-%m', MAX(submitted_date)) FROM claims)",
    ),
    (
        "Which equipment category has the most open maintenance tickets?",
        "SELECT category, COUNT(*) AS open_tickets FROM maintenance_tickets WHERE status = 'open' "
        "GROUP BY category ORDER BY open_tickets DESC LIMIT 1",
    ),
    (
        "What is the total approved amount per insurer?",
        "SELECT insurer, SUM(approved_amount) AS total_approved FROM claims WHERE approved_amount IS NOT NULL "
        "GROUP BY insurer ORDER BY total_approved DESC",
    ),
]

ANSWER_SYSTEM = """You are MedBud, answering an operations question from a database query result.
State the answer plainly in one or two sentences (or a short list if several rows), with the numbers
from the result. Mention that figures come from the 2024 operations data. Do not invent values."""


def generate_sql(question: str) -> str:
    """Step 1: LLM writes the SQL from schema + few-shot examples."""
    shots = "\n\n".join(f"Question: {q}\nSQL: {s}" for q, s in FEW_SHOTS)
    user = f"Schema:\n{schema_summary()}\n\nExamples:\n{shots}\n\nQuestion: {question}\nSQL:"
    return chat(SQL_SYSTEM, user, temperature=0.0, max_tokens=300)


def clean_sql(raw: str) -> str:
    """Step 2: keep only the SQL statement, then enforce the safety rules."""
    text = raw.strip()
    fence = re.search(r"```(?:sql)?\s*(.*?)```", text, re.S | re.I)
    if fence:
        text = fence.group(1)
    text = re.sub(r"^\s*(sql\s*query|sqlquery|sql)\s*:\s*", "", text, flags=re.I).strip()
    statements = [s.strip() for s in text.split(";") if s.strip()]
    if len(statements) != 1:
        raise SQLSafetyError("expected exactly one SQL statement")
    sql = statements[0]
    if not re.match(r"^\s*(select|with)\b", sql, re.I):
        raise SQLSafetyError("only SELECT queries are allowed")
    if FORBIDDEN.search(sql):
        raise SQLSafetyError("query contains a forbidden keyword")
    if not re.search(r"\blimit\b", sql, re.I):
        sql = f"{sql} LIMIT {MAX_ROWS}"
    return sql


def run_sql(sql: str) -> tuple[list[str], list[tuple]]:
    """Step 3a: execute on a read-only connection."""
    con = connect()
    try:
        cur = con.execute(sql)
        columns = [d[0] for d in cur.description] if cur.description else []
        rows = cur.fetchmany(MAX_ROWS)
    finally:
        con.close()
    return columns, rows


def phrase_result(question: str, sql: str, columns: list[str], rows: list[tuple]) -> str:
    """Step 3b: LLM turns the rows into a sentence."""
    table = "\n".join([", ".join(columns)] + [", ".join("" if v is None else str(v) for v in r) for r in rows]) or "(no rows)"
    user = f"Question: {question}\nSQL used: {sql}\nResult ({len(rows)} rows):\n{table}"
    return chat(ANSWER_SYSTEM, user, temperature=0.2, max_tokens=300)


def run_sql_rag(question: str) -> SQLResult:
    sql = clean_sql(generate_sql(question))
    columns, rows = run_sql(sql)
    return SQLResult(answer=phrase_result(question, sql, columns, rows), sql=sql, columns=columns, rows=rows)


def sql_rag_chain(question: str) -> str:
    """The three explicit steps the assignment asks for: translate -> clean -> execute + phrase."""
    raw_sql = generate_sql(question)   # 1. natural language -> SQL (LLM)
    sql = clean_sql(raw_sql)           # 2. extract just the statement, enforce read-only rules
    columns, rows = run_sql(sql)       # 3. execute ...
    return phrase_result(question, sql, columns, rows)  # ... and let the LLM phrase the result
