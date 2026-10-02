"""Build the vector store: parse every document under data/mediassist_data/<collection>/,
embed (dense + BM25), and upsert into Qdrant with the RBAC payload.

Usage:
    python scripts/ingest.py              # create the collection if missing, upsert everything
    python scripts/ingest.py --recreate   # drop and rebuild (use after changing chunking)
    python scripts/ingest.py --only clinical nursing   # subset of collections

Run once before starting the API. First run downloads Docling + embedding models.
"""

from __future__ import annotations

import argparse
import collections
import logging
import time

from medibot.config import COLLECTIONS, DATA_DIR
from medibot.ingest import index
from medibot.ingest.parse import EMBED_WINDOW, ChunkRecord, parse_document

logging.disable(logging.WARNING)
SUFFIXES = {".pdf", ".md"}


def parse_all(collections: list[str]) -> list[ChunkRecord]:
    records: list[ChunkRecord] = []
    for collection in collections:
        folder = DATA_DIR / collection
        for path in sorted(p for p in folder.iterdir() if p.suffix.lower() in SUFFIXES):
            t0 = time.time()
            recs = parse_document(path, collection)
            depth = collections_depth(recs)
            print(f"  {collection}/{path.name:<32} {len(recs):>3} chunks  depth {depth}  max {max(r.n_tokens for r in recs)} tok  ({time.time() - t0:.0f}s)")
            records.extend(recs)
    return records


def collections_depth(recs: list[ChunkRecord]) -> dict[int, int]:
    return dict(sorted(collections.Counter(r.heading_path.count(" > ") + 1 for r in recs).items()))


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--recreate", action="store_true", help="drop the collection and rebuild it")
    ap.add_argument("--only", nargs="+", choices=COLLECTIONS, default=COLLECTIONS)
    args = ap.parse_args()

    print(f"Parsing documents from {DATA_DIR} ...")
    records = parse_all(args.only)

    too_long = [r for r in records if r.n_tokens > EMBED_WINDOW]
    if too_long:  # the embedder would truncate these silently — refuse instead
        raise SystemExit(f"{len(too_long)} chunks exceed {EMBED_WINDOW} tokens; lower CHUNK_MAX_TOKENS. First: {too_long[0].heading_path}")

    types = collections.Counter(r.chunk_type for r in records)
    print(f"\n{len(records)} chunks total  types={dict(types)}")

    index.ensure_collection(recreate=args.recreate)
    t0 = time.time()
    index.upsert_records(records)
    print(f"Upserted into '{index.COLLECTION}' in {time.time() - t0:.0f}s; collection now holds {index.count()} points")
    index.close()


if __name__ == "__main__":
    main()
