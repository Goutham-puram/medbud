"""Show what our parser produces for ONE document, before indexing anything.

Usage:  python scripts/inspect_doc.py [path]        (default: clinical/treatment_protocols.pdf)

Prints heading depth per chunk BEFORE and AFTER the heading-level repair (Docling marks every
PDF heading level 1; we recover levels from font size, see medibot/ingest/parse.py), the
chunk_type counts, token lengths against the embedding window, and a few sample chunks.
"""

import collections
import logging
import sys
from pathlib import Path

logging.disable(logging.WARNING)

from medibot.ingest.parse import EMBED_WINDOW, chunk_document, chunker, convert, repair_heading_levels


def main(path: Path) -> None:
    doc = convert(path)
    _, hc = chunker()
    before = collections.Counter(len(c.meta.headings or []) for c in hc.chunk(dl_doc=doc))
    levels = repair_heading_levels(doc)
    records = chunk_document(doc, path, path.parent.name)

    after = collections.Counter(r.heading_path.count(" > ") + 1 for r in records)
    types = collections.Counter(r.chunk_type for r in records)
    toks = [r.n_tokens for r in records]

    print(f"\n{path}: {len(records)} chunks  access_roles={records[0].access_roles}")
    print("heading depth per chunk  BEFORE repair:", dict(sorted(before.items())))
    print("heading levels recovered (level: count):", dict(sorted(levels.items())))
    print("heading depth per chunk  AFTER repair: ", dict(sorted(after.items())))
    print("chunk_type counts:", dict(types))
    print(f"tokens per embedded text: min {min(toks)}  max {max(toks)}  over {EMBED_WINDOW}: {sum(t > EMBED_WINDOW for t in toks)}")
    for r in records[:3]:
        print(f"\n--- [{r.chunk_type}] {r.heading_path}  ({r.n_tokens} tokens)")
        print(r.text[:350])


if __name__ == "__main__":
    main(Path(sys.argv[1] if len(sys.argv) > 1 else "data/mediassist_data/clinical/treatment_protocols.pdf"))
