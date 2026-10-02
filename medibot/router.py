"""Routing: decide whether a question is answered by SQL RAG (numbers over the database) or by
document RAG, and guess which collection a question is about (for a role-aware refusal).

Routes: "sql" (database analytics), "docs" (document RAG), "chat" (greetings / questions about the
assistant itself, answered without retrieval).
Layer 1 — semantic router (offline, ~10 ms): example questions per route are embedded once; a
query is embedded and compared. "sql" questions form a tight family (counts, totals, trends over
claims and tickets), so the SQL route carries a tuned threshold; "docs" questions are too diverse
to match reliably, so the docs route mostly acts as a competitor that lowers false SQL matches.
Layer 2 — when neither route clears its threshold, one small LLM call decides (logged as such).
"""

from __future__ import annotations

import os
import re
from dataclasses import dataclass
from functools import lru_cache

from semantic_router import Route
from semantic_router.encoders import HuggingFaceEncoder
from semantic_router.routers import SemanticRouter

from medibot.config import DENSE_MODEL
from medibot.rag.llm import LLMError, chat

SQL_THRESHOLD = float(os.getenv("ROUTER_SQL_THRESHOLD", "0.38"))
DOCS_THRESHOLD = float(os.getenv("ROUTER_DOCS_THRESHOLD", "0.30"))
CHAT_THRESHOLD = float(os.getenv("ROUTER_CHAT_THRESHOLD", "0.40"))
TOPIC_THRESHOLD = float(os.getenv("ROUTER_TOPIC_THRESHOLD", "0.30"))
LLM_FALLBACK = os.getenv("ROUTER_LLM_FALLBACK", "1") == "1"

SQL_UTTERANCES = [
    "how many claims were rejected last month", "which equipment category has the most open maintenance tickets",
    "total approved amount per insurer", "average claimed amount for cardiology claims", "count of escalated claims by department",
    "how many maintenance tickets are still open", "which insurer has the highest rejection rate", "number of tickets raised in March",
    "what is the average time to resolve a ticket", "list the top 5 departments by claim amount", "how many claims were submitted in 2024",
    "which campus raised the most tickets", "sum of claimed amounts for cashless claims", "how many tickets are escalated",
    "monthly trend of claim submissions", "percentage of claims approved", "which equipment has the most fault reports",
    "count claims by status", "average approved amount by department", "how many pending claims are there",
    "what is the rejection rate of each insurer", "how many tickets did each campus raise", "total claimed amount in 2024 for neurology",
]
DOCS_UTTERANCES = [
    "what is the adult dose of amoxicillin", "steps for hand hygiene", "how many casual leaves do staff get per year",
    "how do I calibrate the infusion pump", "what documents are needed for pre-authorisation", "what is the diagnostic criteria for type 2 diabetes",
    "what PPE is required for airborne precautions", "what is the dress code", "how to handle a needlestick injury", "what does fault code E-07 mean",
    "what is the procedure for central line care", "which drugs are high-alert medications", "what is the notice period for resignation",
    "how to raise a pre-authorisation enhancement", "what are the danger signs in paediatric fever", "what is the room rent sub-limit",
    "how do I reset my password", "what is the CURB-65 score", "explain the escalation matrix for claims", "what are the alarm defaults on the BM-500 monitor",
    "first-line therapy for community acquired pneumonia", "how many steps are in the ICU admission procedure", "show me the insurance billing codes",
    "what is the preventive maintenance schedule for the x-ray unit", "when is calibration due for the infusion pump according to the manual",
    "maintenance steps for the patient monitor", "who do I escalate to when a claim is rejected", "what is the escalation matrix for claims",
    "how do I appeal a rejected claim", "what are the common claim rejection codes",
]
CHAT_UTTERANCES = [  # greetings and questions about the assistant itself: answered without retrieval
    "hi", "hello", "hey there", "hey", "good morning", "good evening", "how are you", "how are you doing",
    "what can you help me with", "what can you do", "what do you do", "who are you", "what is medbud", "help",
    "how do I use this", "what questions can I ask", "thanks", "thank you", "thanks a lot", "bye", "goodbye",
    "see you", "can you help me", "what are you able to answer", "tell me about yourself",
    "what options do I have for my role", "what am I allowed to see", "what can I access with my role",
    "which documents can I read", "what are my permissions", "what can I ask you", "how does this work",
    "how do I use this assistant", "what is this tool for", "what kind of questions do you answer",
    "can I run reports", "do I have access to billing documents", "what collections can I search",
    "what can a nurse ask", "explain what you can do", "what do you know", "where do your answers come from",
]
TOPIC_UTTERANCES = {
    "clinical": ["treatment protocol for hypertension", "drug formulary dose of metformin", "diagnostic reference ranges for haemoglobin",
                 "first-line antibiotics for pneumonia", "ICD-10 code for dengue", "which drugs need renal dose adjustment"],
    "nursing": ["ICU nursing procedure for central line care", "infection control hand hygiene moments", "ventilator bundle for VAP prevention",
                "IV cannula size for a child", "PPE for airborne precautions", "pressure injury prevention bundle"],
    "billing": ["insurance billing codes for procedures", "claim submission and pre-authorisation documents", "claim rejection codes and appeals",
                "room rent sub-limits by insurer", "cashless claim process steps", "co-pay and exclusions reference"],
    "equipment": ["how to calibrate the infusion pump", "fault code on the patient monitor", "autoclave steriliser cycle selection",
                  "preventive maintenance schedule for the x-ray unit", "equipment commissioning and acceptance testing", "battery replacement procedure"],
    "general": ["leave policy and casual leave entitlement", "staff handbook dress code", "code of conduct and whistleblower protection",
                "how to reset my password", "where is the cafeteria", "notice period for resignation"],
}

# A database question asks for a number over records. Without one of these cues, a "sql" match from the
# semantic layer is almost always a document question that merely mentions claims, tickets or equipment.
SQL_CUES = re.compile(
    r"\b(how many|how much|count|number of|total|sum|average|avg|mean|median|most|least|highest|lowest|"
    r"fewest|maximum|minimum|top \d+|rate|percentage|percent|share|trend|per (month|week|year|insurer|department|"
    r"campus|category|status|equipment)|by (month|insurer|department|campus|category|status)|compare|breakdown|"
    r"distribution|last month|this year|in 20\d\d)\b",
    re.I,
)

FALLBACK_SYSTEM = (
    "Classify the message sent to a hospital staff assistant. Reply with exactly one word:\n"
    "sql  - answering needs counting, totals, averages, rates or trends computed from the claims or maintenance_tickets database\n"
    "docs - answering needs facts, procedures, policies, dosages or instructions written in documents\n"
    "chat - a greeting, thanks, small talk, or a question about the assistant itself (what it can do, how to use it)"
)


@dataclass
class RouteDecision:
    route: str            # "sql" | "docs" | "chat"
    confidence: float | None
    method: str           # "semantic" | "llm" | "default"


@lru_cache(maxsize=1)
def _encoder() -> HuggingFaceEncoder:
    return HuggingFaceEncoder(name=DENSE_MODEL, score_threshold=DOCS_THRESHOLD)


@lru_cache(maxsize=1)
def _intent_router() -> SemanticRouter:
    routes = [
        Route(name="sql", utterances=SQL_UTTERANCES, score_threshold=SQL_THRESHOLD),
        Route(name="docs", utterances=DOCS_UTTERANCES, score_threshold=DOCS_THRESHOLD),
        Route(name="chat", utterances=CHAT_UTTERANCES, score_threshold=CHAT_THRESHOLD),
    ]
    return SemanticRouter(encoder=_encoder(), routes=routes, auto_sync="local")


@lru_cache(maxsize=1)
def _topic_router() -> SemanticRouter:
    routes = [Route(name=c, utterances=u, score_threshold=TOPIC_THRESHOLD) for c, u in TOPIC_UTTERANCES.items()]
    return SemanticRouter(encoder=_encoder(), routes=routes, auto_sync="local")


def warm_up() -> None:
    """Load the encoder and both routers (call at API startup so the first request is not slow)."""
    _intent_router()
    _topic_router()


def classify(question: str) -> RouteDecision:
    choice = _intent_router()(question)
    if choice.name == "sql" and not SQL_CUES.search(question):
        choice.name = None  # mentions the data but asks for no number: let the fallback / default decide
    if choice.name in ("sql", "docs", "chat"):
        return RouteDecision(choice.name, choice.similarity_score, "semantic")
    if LLM_FALLBACK:
        try:
            word = chat(FALLBACK_SYSTEM, question, temperature=0.0, max_tokens=5).strip().lower()
            for name in ("sql", "docs", "chat"):
                if word.startswith(name[:3]):
                    return RouteDecision(name, None, "llm")
        except LLMError:
            pass
    return RouteDecision("docs", None, "default")


def guess_collection(question: str) -> str | None:
    """Best-guess collection for refusal wording only; never touches restricted data."""
    choice = _topic_router()(question)
    return choice.name if choice.name in TOPIC_UTTERANCES else None
