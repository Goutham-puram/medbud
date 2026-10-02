"""Cross-encoder reranking: read (question, chunk) PAIRS together and score relevance jointly.

The hybrid retriever casts a wide net (top-10); only the best few chunks may reach the LLM.
A cross-encoder scores each pair with one transformer pass, unlike the bi-encoder used for
retrieval, which embeds question and chunk separately. Scores are logits: clearly relevant
pairs score well above 0, unrelated ones below 0 — which is what the "not found" rule uses.
"""

from __future__ import annotations

from dataclasses import dataclass

from sentence_transformers import CrossEncoder

from medibot.config import RERANK_MIN_SCORE, RERANK_MODEL, TOP_N
from medibot.retrieval.hybrid import Candidate


@dataclass
class Ranked:
    candidate: Candidate
    score: float


_model: CrossEncoder | None = None


def model() -> CrossEncoder:
    global _model
    if _model is None:
        _model = CrossEncoder(RERANK_MODEL)
    return _model


def rerank(question: str, candidates: list[Candidate], top_n: int = TOP_N) -> list[Ranked]:
    """Score every candidate against the question, return the best top_n (highest first)."""
    if not candidates:
        return []
    scores = model().predict([(question, c.text) for c in candidates])
    ranked = sorted(zip(candidates, scores), key=lambda pair: -float(pair[1]))
    return [Ranked(candidate=c, score=float(s)) for c, s in ranked[:top_n]]


def is_confident(ranked: list[Ranked], min_score: float = RERANK_MIN_SCORE) -> bool:
    """The 'not found' rule: if even the best chunk scores below the threshold, answer with no sources."""
    return bool(ranked) and ranked[0].score >= min_score
