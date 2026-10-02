"""Answer generation for the documents route: only the reranked chunks reach the prompt, and the
answer must cite them. Also holds the role-aware refusal wording used when nothing may be cited.
"""

from __future__ import annotations

from dataclasses import dataclass

import re

from medibot.config import ROLE_COLLECTIONS, SQL_ROLES, roles_for_collection
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
        owners = [ROLE_LABEL[r] + "s" for r in roles_for_collection(topic_collection)]
        owners_txt = ", ".join(owners[:-1]) + f" and {owners[-1]}" if len(owners) > 1 else owners[0]
        text = (
            f"As a {ROLE_LABEL[role]}, you don't have access to {topic_collection} documents "
            f"(restricted to {owners_txt}). I can only answer questions from the {allowed_txt} collections."
        )
    else:
        text = f"I could not find this in the documents available to you ({allowed_txt} collections)."
    return Answer(answer=text, sources=[])


# ---------- small talk: greetings and questions about the assistant, no retrieval ----------

COLLECTION_WORDS = {
    "general": "general staff policies (handbook, leave, code of conduct, FAQs)",
    "clinical": "clinical protocols, the drug formulary and diagnostic references",
    "nursing": "nursing procedures and infection control",
    "billing": "insurance billing codes and claim guides",
    "equipment": "equipment operation and maintenance manuals",
}
_GREETING = re.compile(r"^\s*(hi|hello|hey|hey there|good (morning|afternoon|evening)|greetings)\b", re.I)
_THANKS = re.compile(r"\b(thanks|thank you|cheers|appreciate)\b", re.I)
_BYE = re.compile(r"\b(bye|goodbye|see you|good night)\b", re.I)
_HOW_ARE_YOU = re.compile(r"\bhow are you\b", re.I)

SMALL_TALK_SYSTEM = """You are MedBud, the internal assistant of MediAssist Health Network, replying to small talk.
Reply in one or two friendly sentences. Do not answer medical, policy, billing or equipment questions here;
instead point the user to asking a specific question about the documents they can access. No emojis."""


def _joined(items: list[str]) -> str:
    return ", ".join(items[:-1]) + f" and {items[-1]}" if len(items) > 1 else items[0]


def capabilities(role: str) -> str:
    """Role-aware 'what I can do' paragraph; the same facts the UI's help shows."""
    cols = ROLE_COLLECTIONS[role]
    lines = [f"I'm MedBud, MediAssist's internal assistant. As a {ROLE_LABEL[role]} you can ask me about:"]
    lines += [f"- {c}: {COLLECTION_WORDS[c]}" for c in cols]
    if role in SQL_ROLES:
        lines.append("- numbers from the claims and maintenance databases, e.g. how many claims were rejected last month")
    lines.append("I answer only from those sources and show the passages I used. Name the drug, device, code or policy you mean, one question at a time.")
    return "\n".join(lines)


def small_talk(message: str, role: str) -> Answer:
    """Instant, deterministic replies for the common cases; a short persona reply via the LLM otherwise."""
    if _GREETING.match(message) and len(message.split()) <= 6:
        text = f"Hello! I'm MedBud. As a {ROLE_LABEL[role]} you can ask me about {_joined(ROLE_COLLECTIONS[role])} documents" + \
               (", or for numbers from the claims and maintenance databases." if role in SQL_ROLES else ".") + \
               " What would you like to know?"
    elif _HOW_ARE_YOU.search(message):
        text = "Doing well, thank you. What can I look up for you today?"
    elif _THANKS.search(message) and len(message.split()) <= 8:
        text = "You're welcome. Ask me anything else from your documents whenever you need."
    elif _BYE.search(message) and len(message.split()) <= 6:
        text = "Goodbye, and take care."
    elif re.search(r"\b(what can you|what do you do|who are you|help|how do i use|what questions|able to|about yourself|what is medbud)\b", message, re.I):
        text = capabilities(role)
    else:
        text = chat(SMALL_TALK_SYSTEM, f"The user ({ROLE_LABEL[role]}) says: {message}", temperature=0.5, max_tokens=120)
    return Answer(answer=text, sources=[])
