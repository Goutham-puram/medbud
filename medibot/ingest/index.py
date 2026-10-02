"""Embeddings + Qdrant: one collection with a named dense vector (meaning) and a named sparse
vector (BM25, exact words) per chunk, plus the metadata payload the RBAC filter runs on.

Both vectors are stored at index time so retrieval can query them together in ONE call
(see retrieval/hybrid.py) — the spec forbids two separate searches merged in Python.
"""

from __future__ import annotations

from fastembed import SparseTextEmbedding, TextEmbedding
from qdrant_client import QdrantClient, models

from medibot.config import (
    COLLECTION,
    DENSE_MODEL,
    QDRANT_API_KEY,
    QDRANT_PATH,
    QDRANT_URL,
    SPARSE_MODEL,
)
from medibot.ingest.parse import ChunkRecord

DENSE = "dense"  # named vectors inside the collection
SPARSE = "bm25"

_client: QdrantClient | None = None
_dense: TextEmbedding | None = None
_sparse: SparseTextEmbedding | None = None


def client() -> QdrantClient:
    """Server (Docker / Cloud) by default; embedded file mode when QDRANT_PATH is set."""
    global _client
    if _client is None:
        _client = QdrantClient(path=QDRANT_PATH) if QDRANT_PATH else QdrantClient(url=QDRANT_URL, api_key=QDRANT_API_KEY)
    return _client


def embedders() -> tuple[TextEmbedding, SparseTextEmbedding]:
    global _dense, _sparse
    if _dense is None:
        _dense = TextEmbedding(DENSE_MODEL)
        _sparse = SparseTextEmbedding(SPARSE_MODEL)
    return _dense, _sparse


def dense_dim() -> int:
    dense, _ = embedders()
    return len(next(iter(dense.embed(["dimension probe"]))))


def embed_documents(texts: list[str]) -> tuple[list[list[float]], list[models.SparseVector]]:
    dense, sparse = embedders()
    dv = [v.tolist() for v in dense.embed(texts, batch_size=32)]
    sv = [models.SparseVector(indices=s.indices.tolist(), values=s.values.tolist()) for s in sparse.embed(texts, batch_size=32)]
    return dv, sv


def embed_query(text: str) -> tuple[list[float], models.SparseVector]:
    """Queries use query_embed: BM25 query vectors carry term presence; Qdrant applies IDF (Modifier.IDF)."""
    dense, sparse = embedders()
    dv = next(iter(dense.query_embed(text))).tolist()
    s = next(iter(sparse.query_embed(text)))
    return dv, models.SparseVector(indices=s.indices.tolist(), values=s.values.tolist())


def ensure_collection(recreate: bool = False) -> None:
    c = client()
    if recreate and c.collection_exists(COLLECTION):
        c.delete_collection(COLLECTION)
    if c.collection_exists(COLLECTION):
        return
    c.create_collection(
        collection_name=COLLECTION,
        vectors_config={DENSE: models.VectorParams(size=dense_dim(), distance=models.Distance.COSINE)},
        sparse_vectors_config={SPARSE: models.SparseVectorParams(modifier=models.Modifier.IDF)},
    )
    if not QDRANT_PATH:  # payload indexes exist only on a Qdrant server; embedded mode ignores them
        for field in ("access_roles", "collection", "source_document", "chunk_type"):
            c.create_payload_index(COLLECTION, field_name=field, field_schema=models.PayloadSchemaType.KEYWORD)


def upsert_records(records: list[ChunkRecord], batch_size: int = 64) -> int:
    """Embed and upsert; ids are deterministic (uuid5 of collection/file/index) so re-runs overwrite in place."""
    c = client()
    for start in range(0, len(records), batch_size):
        batch = records[start : start + batch_size]
        dv, sv = embed_documents([r.text for r in batch])
        points = [
            models.PointStruct(id=r.chunk_id, vector={DENSE: dv[i], SPARSE: sv[i]}, payload=r.payload())
            for i, r in enumerate(batch)
        ]
        c.upsert(collection_name=COLLECTION, points=points, wait=True)
    return len(records)


def count() -> int:
    return client().count(COLLECTION, exact=True).count


def close() -> None:
    """Release the client (matters for embedded mode, which holds a lock on its folder)."""
    global _client
    if _client is not None:
        _client.close()
        _client = None
