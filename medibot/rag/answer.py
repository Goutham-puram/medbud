"""Answer generation for the documents route: only the reranked chunks reach the prompt, and the
answer must cite them. Also holds the role-aware refusal wording used when nothing may be cited.
"""

from __future__ import annotations

from dataclasses import dataclass

from medibot.config import ROLE_COLLECTIONS, roles_for_collection
from medibot.rag.llm import chat
from medibot.retrieval.rerank import Ranked

SYSTEM = """You are MedBud, the internal assistant of MediAssist Health Network.
Rules:
1. Answer ONLY from the numbered context passages. If they do not contain the answer, say exactly:
   "I could not find this in the documents available to you." Never guess clinical details.
2. Cite every fact with the passage number in square brackets, e.g. [1] or [2][3].
3. Keep doses, units, codes and numbers exactly as written in the passages.
4. Be concise: a short paragraph or a few bullet points. No preamble."""

ROLE_LABEL = {
    "doctor": "doctor",
    "nurse": "nurse",
    "billing_executive": "billing executive",
    "technician": "technician",
    "admin": "administrator",
}


@dataclass
class Source:
    source_document: str
    section_title: str
    collection: str


@dataclass
class Answer:
    answer: str
    sources: list[Source]


def build_context(ranked: list[Ranked]) -> str:
    """[n] (document > heading path) + the chunk text, one block per reranked chunk."""
    blocks = []
    for n, r in enumerate(ranked, start=1):
        c = r.candidate
        blocks.append(f"[{n}] ({c.source_document} > {c.heading_path})\n{c.text}")
    return "\n\n".join(blocks)


def sources_of(ranked: list[Ranked]) -> list[Source]:
    """Citation list built AFTER retrieval and reranking, so it can only name chunks the role may see."""
    seen: set[tuple[str, str]] = set()
    out: list[Source] = []
    for r in ranked:
        c = r.candidate
        key = (c.source_document, c.section_title)
        if key not in seen:
            seen.add(key)
            out.append(Source(c.source_document, c.section_title, c.collection))
    return out


def generate(question: str, ranked: list[Ranked]) -> Answer:
    user = f"Context passages:\n\n{build_context(ranked)}\n\nQuestion: {question}"
    text = chat(SYSTEM, user, temperature=0.2, max_tokens=600)
    return Answer(answer=text, sources=sources_of(ranked))


def refusal(role: str, topic_collection: str | None) -> Answer:
    """Role-aware 'no' with sources=[] (team guidance, Sep 11).

    topic_collection: the collection the question seems to be about (from the router's topic
    guess), or None when unknown. Never derived from restricted chunks.
    """
    allowed = ROLE_COLLECTIONS[role]
    allowed_txt = ", ".join(allowed[:-1]) + f" and {allowed[-1]}" if len(allowed) > 1 else allowed[0]
    if topic_collection and topic_collection not in allowed:
        owners = [ROLE_LABEL[r] for r in roles_for_collection(topic_collection)]
        owners_txt = ", ".join(owners[:-1]) + f" and {owners[-1]}" if len(owners) > 1 else owners[0]
        text = (
            f"As a {ROLE_LABEL[role]}, you don't have access to {topic_collection} documents "
            f"(restricted to {owners_txt}s). I can only answer questions from the {allowed_txt} collections."
        )
    else:
        text = f"I could not find this in the documents available to you ({allowed_txt} collections)."
    return Answer(answer=text, sources=[])
