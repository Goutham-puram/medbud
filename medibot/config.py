"""Central settings. Every tunable comes from the environment (.env); nothing is hard-coded elsewhere."""

import os
import secrets
from pathlib import Path

from dotenv import load_dotenv

load_dotenv()

ROOT = Path(__file__).resolve().parents[1]
DATA_DIR = Path(os.getenv("MEDIBOT_DATA_DIR", ROOT / "data" / "mediassist_data"))
DB_PATH = Path(os.getenv("MEDIBOT_DB_PATH", DATA_DIR / "db" / "mediassist.db"))

# vector store
QDRANT_URL = os.getenv("QDRANT_URL", "http://localhost:6333")
QDRANT_API_KEY = os.getenv("QDRANT_API_KEY") or None
QDRANT_PATH = os.getenv("QDRANT_PATH") or None  # set -> embedded file mode instead of a server
COLLECTION = os.getenv("QDRANT_COLLECTION", "medibot_chunks")

# models
DENSE_MODEL = os.getenv("DENSE_MODEL", "sentence-transformers/all-MiniLM-L6-v2")
SPARSE_MODEL = os.getenv("SPARSE_MODEL", "Qdrant/bm25")
RERANK_MODEL = os.getenv("RERANK_MODEL", "cross-encoder/ms-marco-MiniLM-L-6-v2")
MAX_TOKENS = int(os.getenv("CHUNK_MAX_TOKENS", "200"))  # embedded text must stay under the 256-token window

# llm
GROQ_API_KEY = os.getenv("GROQ_API_KEY", "")
GROQ_MODEL = os.getenv("GROQ_MODEL", "openai/gpt-oss-120b")

# auth — HS256 wants a secret of at least 32 bytes
JWT_SECRET = os.getenv("JWT_SECRET", "")
if len(JWT_SECRET) < 32:  # a random per-process secret keeps dev safe (tokens stop working on restart)
    print('WARNING: JWT_SECRET missing or shorter than 32 characters; using a random secret for this process. '
          'Set one in .env, e.g.  python -c "import secrets; print(secrets.token_hex(32))"')
    JWT_SECRET = secrets.token_hex(32)
JWT_HOURS = int(os.getenv("JWT_HOURS", "8"))

# retrieval knobs
CANDIDATES_K = int(os.getenv("CANDIDATES_K", "10"))
TOP_N = int(os.getenv("TOP_N", "3"))
RERANK_MIN_SCORE = float(os.getenv("RERANK_MIN_SCORE", "-3.0"))  # cross-encoder logit; on the eval set denials <= -4.0, answerable >= +3.2
NEARBY_TOPICS_FLOOR = float(os.getenv("NEARBY_TOPICS_FLOOR", "-10.5"))  # below this the question is unrelated to any document: suggest nothing

# RBAC — two axes: documents by department, SQL by function (team clarification, Sep 5)
COLLECTIONS = ["general", "clinical", "nursing", "billing", "equipment"]
ROLE_COLLECTIONS: dict[str, list[str]] = {
    "doctor": ["clinical", "nursing", "general"],
    "nurse": ["nursing", "general"],
    "billing_executive": ["billing", "general"],
    "technician": ["equipment", "general"],
    "admin": COLLECTIONS,
}
SQL_ROLES = {"billing_executive", "admin"}


def roles_for_collection(collection: str) -> list[str]:
    """Inverse of ROLE_COLLECTIONS: the roles allowed to read a collection -> the chunk's access_roles."""
    return [role for role, cols in ROLE_COLLECTIONS.items() if collection in cols]
