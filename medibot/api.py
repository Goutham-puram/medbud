"""FastAPI backend.

POST /login               username + password -> signed JWT carrying the role
POST /chat                Bearer token; question -> answer + sources + retrieval_type + role (+ instrumentation)
GET  /me                  Bearer token; the token's user/role/collections (session restore for clients)
GET  /collections/{role}  collections a role may read
GET  /health              liveness + vector-store count

/chat logic: role from the token -> router (sql | docs | chat) -> greetings and "what can you do"
answered directly (no retrieval) -> SQL RAG if the role has analytics rights,
else hybrid retrieval with the role filter inside the Qdrant query -> cross-encoder top-3 ->
confident? answer with citations : role-aware refusal with sources=[].
"""

from __future__ import annotations

import time
from contextlib import asynccontextmanager

from fastapi import Depends, FastAPI, HTTPException
from fastapi.security import HTTPAuthorizationCredentials, HTTPBearer
from typing import Literal

from pydantic import BaseModel, Field

from medibot import router
from medibot.auth import AuthError, authenticate, issue_token, verify_token
from medibot.config import GROQ_MODEL, NEARBY_TOPICS_FLOOR, ROLE_COLLECTIONS, SQL_ROLES
from medibot.ingest import index
from medibot.obs import Timer, log_event, new_request_id
from medibot.rag import answer as answer_mod
from medibot.rag.llm import LLMError
from medibot.retrieval import hybrid, rerank
from medibot.sql_rag import SQLSafetyError, run_sql_rag



@asynccontextmanager
async def lifespan(_: FastAPI):
    """Load models once at startup so the first request is not the slow one."""
    index.embedders()
    rerank.model()
    router.warm_up()
    yield
    index.close()


app = FastAPI(title="MedBud API", version="0.1.0", lifespan=lifespan,
              description="Role-aware hybrid RAG + SQL RAG for MediAssist Health Network")
bearer = HTTPBearer(auto_error=False)


# ---------- schemas ----------

class LoginRequest(BaseModel):
    username: str = Field(min_length=1, max_length=64)
    password: str = Field(min_length=1, max_length=128)


class LoginResponse(BaseModel):
    access_token: str
    token_type: str = "bearer"
    username: str
    role: str
    collections: list[str]
    sql_access: bool


class ChatRequest(BaseModel):
    question: str = Field(min_length=2, max_length=1000)
    role: str | None = Field(default=None, description="Ignored: the role always comes from the token. Logged if it differs.")


class SourceOut(BaseModel):
    source_document: str
    section_title: str
    collection: str


class ChatResponse(BaseModel):
    answer: str
    sources: list[SourceOut]
    retrieval_type: Literal["hybrid_rag", "sql_rag", "direct"]  # "direct" = no retrieval (greetings / about the assistant)
    role: str
    # instrumentation (Assignment 3 reads these)
    request_id: str
    route: str
    route_confidence: float | None
    route_method: str
    denied: bool
    rerank_scores: list[float]
    sql: str | None = None
    latency_ms: int


# ---------- helpers ----------

def current_claims(creds: HTTPAuthorizationCredentials | None = Depends(bearer)) -> dict:
    if creds is None:
        raise HTTPException(status_code=401, detail="missing bearer token")
    try:
        return verify_token(creds.credentials)
    except AuthError as exc:
        raise HTTPException(status_code=401, detail=str(exc)) from exc


# ---------- endpoints ----------

@app.get("/health")
def health() -> dict:
    try:
        points = index.count()
        store = "ok"
    except Exception as exc:  # the store being down is exactly what /health must report
        points, store = 0, f"error: {type(exc).__name__}"
    return {"status": "ok" if store == "ok" else "degraded", "vector_store": store, "chunks": points, "llm": GROQ_MODEL}


@app.post("/login", response_model=LoginResponse)
def login(body: LoginRequest) -> LoginResponse:
    try:
        role = authenticate(body.username, body.password)
    except AuthError as exc:
        log_event(event="login_failed", username=body.username)
        raise HTTPException(status_code=401, detail=str(exc)) from exc
    log_event(event="login", username=body.username, role=role)
    return LoginResponse(
        access_token=issue_token(body.username, role),
        username=body.username,
        role=role,
        collections=ROLE_COLLECTIONS[role],
        sql_access=role in SQL_ROLES,
    )


@app.get("/me")
def me(claims: dict = Depends(current_claims)) -> dict:
    """Who am I, according to my token. Lets a client restore its session after a page reload."""
    role = claims["role"]
    return {"username": claims.get("sub"), "role": role, "collections": ROLE_COLLECTIONS[role],
            "sql_access": role in SQL_ROLES, "expires_at": claims.get("exp")}


@app.get("/collections/{role}")
def collections(role: str) -> dict:
    if role not in ROLE_COLLECTIONS:
        raise HTTPException(status_code=404, detail=f"unknown role '{role}'")
    return {"role": role, "collections": ROLE_COLLECTIONS[role], "sql_access": role in SQL_ROLES}


@app.post("/chat", response_model=ChatResponse)
def chat(body: ChatRequest, claims: dict = Depends(current_claims)) -> ChatResponse:
    role = claims["role"]                         # the ONLY source of truth for the role
    request_id = new_request_id()
    timings: dict = {}
    t_start = time.perf_counter()
    if body.role and body.role != role:           # a spoof attempt is worth a log line, nothing more
        log_event(event="role_mismatch", request_id=request_id, token_role=role, claimed_role=body.role)

    with Timer(timings, "route"):
        decision = router.classify(body.question)

    denied, sql, rerank_scores, sources = False, None, [], []
    if decision.route == "chat":
        retrieval_type = "direct"
        try:
            with Timer(timings, "small_talk"):
                text = answer_mod.small_talk(body.question, role).answer
        except LLMError as exc:
            raise HTTPException(status_code=503, detail=str(exc)) from exc
    elif decision.route == "sql":
        retrieval_type = "sql_rag"
        if role not in SQL_ROLES:
            denied = True
            allowed = ", ".join(sorted(SQL_ROLES))
            text = (f"As a {answer_mod.ROLE_LABEL[role]}, you don't have access to analytics over the claims and "
                    f"maintenance databases (restricted to {allowed}). I can answer questions from the "
                    f"{', '.join(ROLE_COLLECTIONS[role])} documents.")
        else:
            try:
                with Timer(timings, "sql_rag"):
                    result = run_sql_rag(body.question)
                text, sql = result.answer, result.sql
            except SQLSafetyError as exc:
                raise HTTPException(status_code=400, detail=f"could not form a safe query: {exc}") from exc
            except LLMError as exc:
                raise HTTPException(status_code=503, detail=str(exc)) from exc
            except Exception as exc:  # sqlite errors on a bad query -> client error, logged with the SQL
                log_event(event="sql_error", request_id=request_id, error=repr(exc))
                raise HTTPException(status_code=400, detail="the generated query could not be executed") from exc
    else:
        retrieval_type = "hybrid_rag"
        with Timer(timings, "retrieval"):
            candidates = hybrid.search(body.question, role)
        with Timer(timings, "rerank"):
            ranked = rerank.rerank(body.question, candidates)
        rerank_scores = [round(r.score, 3) for r in ranked]
        if rerank.is_confident(ranked):
            try:
                with Timer(timings, "generate"):
                    result = answer_mod.generate(body.question, ranked)
            except LLMError as exc:
                raise HTTPException(status_code=503, detail=str(exc)) from exc
            if result.not_found:  # passages were plausible but did not contain the answer -> same path as a refusal
                denied = True
                text = answer_mod.refusal(role, None, ranked).answer
            else:
                text, sources = result.answer, result.sources
        else:
            denied = True
            related = ranked if ranked and ranked[0].score > NEARBY_TOPICS_FLOOR else []  # near-miss: offer topics
            text = answer_mod.refusal(role, router.guess_collection(body.question), related).answer

    latency_ms = round((time.perf_counter() - t_start) * 1000)
    log_event(
        event="chat", request_id=request_id, user=claims.get("sub"), role=role, route=decision.route,
        route_confidence=decision.confidence, route_method=decision.method, retrieval_type=retrieval_type,
        denied=denied, rerank_scores=rerank_scores, n_sources=len(sources), sql=sql,
        question=body.question[:200], latency_ms=latency_ms, **timings,
    )
    return ChatResponse(
        answer=text,
        sources=[SourceOut(**s.__dict__) for s in sources],
        retrieval_type=retrieval_type,
        role=role,
        request_id=request_id,
        route=decision.route,
        route_confidence=decision.confidence,
        route_method=decision.method,
        denied=denied,
        rerank_scores=rerank_scores,
        sql=sql,
        latency_ms=latency_ms,
    )
