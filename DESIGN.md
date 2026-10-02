# MediBot — design v1 (2026-10-01) · ACCEPTED by GP 10-01 · repo github.com/Goutham-puram/medbud · author GTP

Read **Part A** (2 minutes). Part B is reference for when we build — you don't need it now.

---

# Part A — the design in plain words

## A1. What we are building, in 8 lines

1. A **one-time script** turns the 12 documents into chunks and stores them in **Qdrant** (vector database).
2. Every chunk is stored with **two vectors** — one for meaning (dense), one for exact words (BM25) — plus **who may see it** (`access_roles`).
3. A user **logs in** and gets a **signed token** that carries their role. The server trusts only the token, never what the browser sends.
4. For every question a **router** decides: *numbers question* → **SQL RAG**; *document question* → **Hybrid RAG**.
5. Hybrid RAG asks Qdrant **one** question that searches both vectors at once, **with the role filter inside the database query**, and gets 10 candidates.
6. A **cross-encoder** reads (question + chunk) pairs and keeps the **best 3**; if even the best is weak, the user gets a polite, role-aware "no access / not found" and **no sources**.
7. The **LLM (Groq)** answers from those 3 chunks only, with citations.
8. The reply carries `answer`, `sources`, `retrieval_type`, `role` — plus timing and scores, so Assignment 3 can measure it later.

## A2. Picture

```
            ┌──────────── ONE-TIME: scripts/ingest.py ─────────────┐
            │ PDF/MD ─► Docling (structure) ─► HybridChunker        │
            │        ─► "Heading > Sub" + text  ─► dense + BM25     │
            │        ─► Qdrant  (payload: collection, access_roles, │
            │                    section_title, chunk_type, ...)    │
            └───────────────────────────────────────────────────────┘

  user ─► /login ─► JWT {role} ─► /chat ─► ROUTER ─┬─ "sql"  ─► allowed? ─► sql_rag_chain() ─► answer
                                                   │                └ no ─► refusal, sources=[]
                                                   └─ "docs" ─► Qdrant hybrid query  (role filter INSIDE)
                                                                 ─► 10 candidates ─► cross-encoder ─► top 3
                                                                 ─► weak? refusal, sources=[]
                                                                 ─► Groq answer + citations
```

## A3. Who sees what (RBAC)

| Role | Documents (by department) | SQL (by function) |
|---|---|---|
| doctor | clinical + nursing + general | no |
| nurse | nursing + general | no |
| billing_executive | billing + general | **yes** |
| technician | equipment + general | no |
| admin | everything | **yes** |

The folder name of each document *is* its collection, so `access_roles` is derived, never hand-typed.
Refusal wording: *"As a nurse, you don't have access to billing documents. I can answer from the nursing and general collections."*

## A4. The decisions (and the one-line reason)

| # | Decision | Choice | Why |
|---|---|---|---|
| 1 | Frontend | Streamlit calling FastAPI | Team allows it; you know it; API stays clean for Next.js later |
| 2 | Vector DB | Qdrant in Docker (`docker compose up qdrant`) | Ingest script and API can both connect; one command for the grader |
| 3 | Where code runs | API + UI on your Mac via `uv`; only Qdrant in Docker | Smallest setup; full containerisation is a stretch goal |
| 4 | Parsing + chunking | Docling → heading-level repair (font-size clusters) → `HybridChunker(max_tokens=200)`; full breadcrumb in the embedded text | Team: HybridChunker not RPP; Docling flattens PDF headings (verified 10-01); 200 keeps every chunk ≤ 229 tokens, under MiniLM's 256 |
| 5 | Embeddings | fastembed: `all-MiniLM-L6-v2` (dense) + `Qdrant/bm25` (sparse) | One library for both vectors; Qdrant's own; no torch at query time |
| 6 | Fusion | Qdrant RRF, inside one `query_points` call | Dense and BM25 scores aren't comparable; spec wants one query |
| 7 | Reranker | `cross-encoder/ms-marco-MiniLM-L-6-v2` (sentence-transformers), 10 → 3 | The class-taught model; name matches the rubric |
| 8 | "Not found" rule | reranker best score < threshold → refusal, `sources=[]` | Team: never cite weak chunks; threshold tuned on the eval set |
| 9 | Router | semantic-router (offline) + Groq fallback if unsure | Team recommends semantic router; fallback catches odd phrasings |
| 10 | SQL RAG | plain function, 3 steps; read-only DB; SELECT-only; LIMIT 50 | Spec wording; safety against "DROP TABLE" prompts |
| 11 | Auth | `/login` → JWT (HS256, 8 h); passwords hashed; role read from token | The peer rejection was exactly "role from the client" |
| 12 | LLM | Groq `openai/gpt-oss-120b`, env-overridable, temp 0.2 | Same as fit-plan.ai; free tier 8k tok/min → keep prompts small |
| 13 | Code shape | Python package `medibot/` with small modules | Capstone options 03/05 reuse the same parts |
| 14 | Jev | not in the graded path | One-week-old early-access API; optional A/B later |

## A5. What the grader will see (how we prove it)

- **RBAC (25%)**: at least 3 adversarial prompts per restricted role, with screenshots **and** the log line showing the filter and `sources=[]`.
- **Hybrid + rerank (20%)**: a small script scores dense-only vs hybrid vs hybrid+rerank on our 15+ labelled questions (recall@3) → table in README.
- **SQL RAG (15%)**: 4+ analytical questions with the generated SQL shown.
- **Ingestion (20%)**: sample chunk printouts showing breadcrumb + all 5 metadata fields.
- Fresh-clone run + evaluator-persona review before submitting.

## A6. Risks we already know about

- ~~Docling may flatten PDF heading levels~~ CONFIRMED 10-01 (all 11 PDFs: every heading level 1). FIXED in parse.py: heading bbox height = font size → cluster into levels → chunker builds the real path. Result over all 12 docs: 295 chunks, depth 2-4, max 229 tokens.
- Docling downloads its models on first run (minutes) → run ingest once, early.
- Thresholds (reranker, router) are placeholders until the eval set exists (queue item 8).
- Groq free tier rate limit → 3 chunks per prompt, no retries in loops.

**Decided 10-01:** design v1 accepted; repo `Goutham-puram/medbud` (folder stays `medibot`); author name GTP; GTPai account left unused.

---

# Part B — reference for implementation (skip for now)

## B1. Folder layout

```
medibot/                       git repo (product name TBD)
  README.md  DESIGN.md  docker-compose.yml  .env.example  pyproject.toml / requirements.txt
  data/mediassist_data/        copy of the dataset (12 docs + mediassist.db) → repo is self-contained
  medibot/                     python package
    config.py                  env: QDRANT_URL, QDRANT_API_KEY, GROQ_API_KEY, GROQ_MODEL, JWT_SECRET, thresholds
    auth.py                    demo users (pbkdf2 hashes), JWT issue/verify, ROLE_COLLECTIONS, SQL_ROLES
    ingest/parse.py            Docling convert → HybridChunker → contextualize() → metadata
    ingest/index.py            fastembed dense+sparse → Qdrant collection (named vectors dense/bm25) → upsert
    retrieval/hybrid.py        ONE query_points: prefetch[dense, bm25] + FusionQuery(RRF) + access_roles filter
    retrieval/rerank.py        CrossEncoder top-3 + RERANK_MIN_SCORE
    rag/answer.py              prompt (context-only, cite [n]) → Groq → answer + sources
    sql_rag.py                 sql_rag_chain(question) -> str  (LLM→SQL, clean_sql, run ro, LLM phrases)
    router.py                  Route("sql"), Route("docs") + HuggingFaceEncoder(all-MiniLM) + Groq fallback
    api.py                     /login /chat /collections/{role} /health
    obs.py                     request_id, timers, one JSON log line per request
  scripts/ingest.py            CLI ingest;  scripts/eval_retrieval.py  dense vs hybrid vs hybrid+rerank
  ui/app.py                    Streamlit: login, role badge + collections, chat, retrieval-type label, citations
  evals/questions.jsonl        15+ labelled Qs incl. adversarial (shared with A3)
  tests/                       rbac_adversarial, sql_clean, router, api smoke
```

## B2. Chunk payload (what every Qdrant point carries)

`source_document` (filename) · `collection` (folder) · `access_roles` (list) · `section_title` (last heading) ·
`chunk_type` (table / code / heading / text, from Docling item labels) · `heading_path` (full breadcrumb) ·
`text` · `chunk_index` · `doc_hash` (stable ids → re-ingest only changed docs).

## B3. `/chat` response

Required: `answer`, `sources[{source_document, section_title, collection}]`, `retrieval_type` (`hybrid_rag` | `sql_rag`), `role`.
Extra (instrumentation for A3): `request_id`, `route`, `route_confidence`, `denied`, `rerank_scores`, `latency_ms`.

## B4. SQL safety rules

Read-only SQLite connection (`mode=ro`); only one statement; must start with SELECT; LIMIT 50 appended; schema and
3 few-shot examples in the prompt; table/column names read from `PRAGMA table_info` at startup, never hard-coded.

## B5. Versions (verified against PyPI / installed on 10-01)

qdrant-client 1.19.1 · fastembed 0.8.1 · docling 2.132.0 (docling-core 2.99 chunker) · sentence-transformers 6.1.0 ·
semantic-router 0.1.16 · PyJWT 2.15.1 · fastapi 0.142.2 · uvicorn 0.54.0 · streamlit 1.64.0 · Python 3.12 ·
Qdrant image `qdrant/qdrant`. Alternative reranker if image size matters later: fastembed `Xenova/ms-marco-MiniLM-L-6-v2` (same weights, ONNX).

## B6. Verified API shapes (so implementation doesn't guess)

- Qdrant: `query_points(collection, prefetch=[Prefetch(query=dense, using="dense", filter=F, limit=10), Prefetch(query=SparseVector, using="bm25", filter=F, limit=10)], query=FusionQuery(fusion=Fusion.RRF), query_filter=F, limit=10)`; sparse config `SparseVectorParams(modifier=Modifier.IDF)`; filter `FieldCondition(key="access_roles", match=MatchAny(any=[role]))`.
- Docling: `DocumentConverter().convert(path).document`; `HybridChunker(tokenizer=HuggingFaceTokenizer(AutoTokenizer(all-MiniLM), max_tokens=256), merge_peers=True)`; `chunker.contextualize(chunk)`; `chunk.meta.headings`, `chunk.meta.doc_items[i].label` (table / code / section_header / text ...).
- semantic-router: `SemanticRouter(encoder=HuggingFaceEncoder(name=...), routes=[...], auto_sync="local")`; `sr(query).name`; `sr.fit(X, y)` for thresholds.
- PyJWT: `jwt.encode(payload, secret, algorithm="HS256")` / `jwt.decode(token, secret, algorithms=["HS256"])`.
- CrossEncoder: `CrossEncoder(model).predict([(q, text), ...])` → scores; `.rank(q, docs, top_k=3)` also available.
