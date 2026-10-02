"""Evaluate MedBud on evals/questions.jsonl and write a markdown report.

    python scripts/evaluate.py                 # offline: retrieval quality, RBAC, router (no LLM calls)
    python scripts/evaluate.py --e2e           # also call the running API for every case (needs API + Groq key)

Offline mode measures, over the document questions, whether the expected passage is found by
dense-only search, by hybrid search, and by hybrid + cross-encoder reranking (hit@1, hit@3, MRR) —
the "demonstrably better than dense-only" evidence. It also checks every adversarial case: no
candidate from a restricted collection, the pipeline refuses, and the router picks the right route.
End-to-end mode additionally checks the final answers (must_contain keywords, denied flag, sources).
"""

from __future__ import annotations

import argparse
import json
import logging
import os
import re
import time
from pathlib import Path

logging.disable(logging.WARNING)

from medibot import router
from medibot.config import ROLE_COLLECTIONS, SQL_ROLES
from medibot.ingest import index
from medibot.retrieval import hybrid, rerank

ROOT = Path(__file__).resolve().parents[1]
CASES = ROOT / "evals" / "questions.jsonl"
PASSWORDS = {"doctor": ("dr.mehta", "doctor123"), "nurse": ("nurse.priya", "nurse123"),
             "billing_executive": ("billing.ravi", "billing123"), "technician": ("tech.anand", "tech123"),
             "admin": ("admin.sys", "admin123")}


def normalise(text: str) -> str:
    """Make keyword checks robust to markdown emphasis and typographic hyphens the LLM likes to use."""
    text = re.sub(r"[\u2010\u2011\u2012\u2013\u2014\u2212]", "-", text)   # ‐ ‑ ‒ – — − -> -
    text = re.sub(r"[*_`]", "", text)                                       # **C**onfusion -> Confusion
    return re.sub(r"\s+", " ", text).lower()


def load_cases() -> list[dict]:
    return [json.loads(line) for line in CASES.read_text().splitlines() if line.strip()]


def is_hit(candidate, case) -> bool:
    return (candidate.source_document == case["expected_source_document"]
            and case["expected_section"].lower() in candidate.heading_path.lower())


def rank_of(cands, case) -> int | None:
    for i, c in enumerate(cands, 1):
        if is_hit(c, case):
            return i
    return None


def retrieval_eval(cases: list[dict]) -> tuple[str, dict]:
    docs = [c for c in cases if c["kind"] == "docs"]
    stats = {m: {"hit1": 0, "hit3": 0, "rr": 0.0} for m in ("dense_only", "hybrid", "hybrid_rerank")}
    detail = []
    for case in docs:
        q, role = case["question"], case["role"]
        dense = hybrid.search_dense_only(q, role)
        hyb = hybrid.search(q, role)
        rr = [r.candidate for r in rerank.rerank(q, hyb, top_n=3)]
        ranks = {"dense_only": rank_of(dense, case), "hybrid": rank_of(hyb, case), "hybrid_rerank": rank_of(rr, case)}
        for m, r in ranks.items():
            if r:
                stats[m]["hit1"] += r == 1
                stats[m]["hit3"] += r <= 3
                stats[m]["rr"] += 1 / r
        detail.append((case["id"], ranks, q))
    n = len(docs)
    lines = ["| Method | hit@1 | hit@3 | MRR |", "|---|---|---|---|"]
    for m, s in stats.items():
        lines.append(f"| {m} | {s['hit1']}/{n} ({s['hit1']/n:.0%}) | {s['hit3']}/{n} ({s['hit3']/n:.0%}) | {s['rr']/n:.2f} |")
    lines += ["", "Rank of the expected passage per question (None = not in the top-10 / top-3):", "",
              "| id | dense | hybrid | hybrid+rerank | question |", "|---|---|---|---|---|"]
    for cid, ranks, q in detail:
        lines.append(f"| {cid} | {ranks['dense_only']} | {ranks['hybrid']} | {ranks['hybrid_rerank']} | {q} |")
    return "\n".join(lines), stats


def rbac_eval(cases: list[dict]) -> str:
    lines = ["| id | role | question | restricted | leaked? | refused? | route |", "|---|---|---|---|---|---|---|"]
    ok = 0
    denied = [c for c in cases if c["kind"] == "denied"]
    for case in denied:
        q, role = case["question"], case["role"]
        decision = router.classify(q)
        if decision.route == "sql":
            leaked, refused = False, role not in SQL_ROLES
        else:
            cands = hybrid.search(q, role)
            leaked = any(c.collection not in ROLE_COLLECTIONS[role] for c in cands)
            ranked = rerank.rerank(q, cands)
            refused = not rerank.is_confident(ranked)
        passed = (not leaked) and refused
        ok += passed
        lines.append(f"| {case['id']} | {role} | {q} | {case.get('restricted_collection') or '-'} | "
                     f"{'YES' if leaked else 'no'} | {'yes' if refused else 'NO'} | {decision.route} ({decision.method}) |")
    lines.append(f"\n**{ok}/{len(denied)} adversarial / off-topic cases handled correctly** (no restricted chunk retrieved, answer refused).")
    return "\n".join(lines)


def router_eval(cases: list[dict]) -> str:
    rows, ok, n = [], 0, 0
    for case in cases:
        want = case.get("expected_route")
        if not want:
            continue
        d = router.classify(case["question"])
        n += 1
        ok += d.route == want
        if d.route != want:
            rows.append(f"| {case['id']} | {case['question']} | {want} | {d.route} ({d.method}) |")
    out = [f"Router agreement with labels: **{ok}/{n}** (LLM fallback {'on' if router.LLM_FALLBACK else 'off'})."]
    if rows:
        out += ["", "| id | question | expected | got |", "|---|---|---|---|"] + rows
    return "\n".join(out)


def e2e_eval(cases: list[dict], api: str) -> str:
    import httpx
    tokens = {}
    for role, (user, pw) in PASSWORDS.items():
        tokens[role] = httpx.post(f"{api}/login", json={"username": user, "password": pw}, timeout=30).json()["access_token"]
    lines = ["| id | role | type | denied | sources | keywords | ms | answer (first 90 chars) |", "|---|---|---|---|---|---|---|---|"]
    ok = 0
    for case in cases:
        r = httpx.post(f"{api}/chat", json={"question": case["question"]},
                       headers={"Authorization": f"Bearer {tokens[case['role']]}"}, timeout=120)
        if r.status_code != 200:
            lines.append(f"| {case['id']} | {case['role']} | ERROR {r.status_code} | | | | | {r.text[:90]} |")
            continue
        b = r.json()
        kw_ok = all(normalise(k) in normalise(b["answer"]) for k in case.get("must_contain", []))
        denied_ok = b["denied"] == case["expect_denied"]
        sources_ok = (b["sources"] == []) if case["expect_denied"] else True
        passed = kw_ok and denied_ok and sources_ok
        ok += passed
        lines.append(f"| {case['id']} | {case['role']} | {b['retrieval_type']} | {b['denied']} | {len(b['sources'])} | "
                     f"{'ok' if kw_ok else 'MISS'} | {b['latency_ms']} | {b['answer'][:90].replace(chr(10), ' ').replace('|', '/')} |")
        time.sleep(2)  # free tier: ~8k tokens per minute; a document answer costs ~1.2k
    lines.append(f"\n**{ok}/{len(cases)} end-to-end cases passed** (expected keywords present, denied flag and empty sources as labelled).")
    return "\n".join(lines)


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--e2e", action="store_true", help="also run every case through the live API")
    ap.add_argument("--api", default=os.getenv("MEDIBOT_API_URL", "http://localhost:8000"))
    args = ap.parse_args()
    cases = load_cases()
    out = [f"# MedBud evaluation — {time.strftime('%Y-%m-%d %H:%M')}", "",
           f"{len(cases)} labelled cases in `evals/questions.jsonl` "
           f"({sum(c['kind']=='docs' for c in cases)} document, {sum(c['kind']=='sql' for c in cases)} SQL, "
           f"{sum(c['kind']=='denied' for c in cases)} adversarial/off-topic, {sum(c['kind']=='chat' for c in cases)} small talk).", ""]
    out += ["## Retrieval quality (document questions)", "", retrieval_eval(cases)[0], ""]
    out += ["## RBAC — adversarial and off-topic", "", rbac_eval(cases), ""]
    out += ["## Router", "", router_eval(cases), ""]
    if args.e2e:
        out += ["## End-to-end through the API", "", e2e_eval(cases, args.api), ""]
    report = "\n".join(out)
    path = ROOT / "evals" / ("results_e2e.md" if args.e2e else "results_offline.md")
    path.write_text(report)
    print(report)
    print(f"\nwritten to {path}")
    index.close()


if __name__ == "__main__":
    main()
