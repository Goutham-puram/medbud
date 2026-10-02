"""Hybrid retrieval: ONE Qdrant query that searches the dense and the BM25 vectors together,
fuses them by rank (RRF), and applies the role filter INSIDE the query — so chunks a role may
not see never leave the database.

Why RRF: cosine scores (0..1) and BM25 scores (unbounded) are not comparable, so results are
merged by rank, not by score (team clarification, Sep 5).
"""

from __future__ import annotations

from dataclasses import dataclass

from qdrant_client import models

from medibot.config import CANDIDATES_K, COLLECTION
from medibot.ingest.index import DENSE, SPARSE, client, embed_query


@dataclass
class Candidate:
    id: str
    score: float
    text: str
    source_document: str
    section_title: str
    collection: str
    chunk_type: str
    heading_path: str


def role_filter(role: str) -> models.Filter:
    """RBAC: the chunk's access_roles list must contain the caller's role (derived from the JWT)."""
    return models.Filter(must=[models.FieldCondition(key="access_roles", match=models.MatchAny(any=[role]))])


def _to_candidates(points) -> list[Candidate]:
    return [
        Candidate(
            id=str(p.id),
            score=float(p.score),
            text=p.payload["text"],
            source_document=p.payload["source_document"],
            section_title=p.payload["section_title"],
            collection=p.payload["collection"],
            chunk_type=p.payload["chunk_type"],
            heading_path=p.payload["heading_path"],
        )
        for p in points
    ]


def search(question: str, role: str, k: int = CANDIDATES_K) -> list[Candidate]:
    """Dense + BM25 prefetch (each role-filtered) -> RRF fusion -> top-k candidates for the reranker."""
    dv, sv = embed_query(question)
    flt = role_filter(role)
    res = client().query_points(
        collection_name=COLLECTION,
        prefetch=[
            models.Prefetch(query=dv, using=DENSE, filter=flt, limit=k),
            models.Prefetch(query=sv, using=SPARSE, filter=flt, limit=k),
        ],
        query=models.FusionQuery(fusion=models.Fusion.RRF),
        query_filter=flt,
        limit=k,
        with_payload=True,
    )
    return _to_candidates(res.points)


def search_dense_only(question: str, role: str, k: int = CANDIDATES_K) -> list[Candidate]:
    """Baseline for the README comparison (dense-only vs hybrid vs hybrid+rerank). Same filter."""
    dv, _ = embed_query(question)
    res = client().query_points(
        collection_name=COLLECTION, query=dv, using=DENSE, query_filter=role_filter(role), limit=k, with_payload=True
    )
    return _to_candidates(res.points)
