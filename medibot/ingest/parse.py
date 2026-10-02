"""Document parsing: Docling -> heading-level repair -> HybridChunker -> chunk records with metadata.

Why the repair step: Docling's PDF pipeline marks EVERY heading as level 1, so the chunker's
breadcrumb would be a single heading ("Diagnostic criteria") with no section above it. Each
heading's bounding-box height tracks its font size, so we cluster heights into levels, write
them back onto the SectionHeaderItems, and let the chunker build the full path itself
("Standard Treatment Protocols > A. Type 2 Diabetes Mellitus > Diagnostic criteria").
Markdown documents already carry real levels from Docling's markdown backend and are left alone.
"""

from __future__ import annotations

import hashlib
import uuid
from dataclasses import asdict, dataclass
from pathlib import Path

from docling.document_converter import DocumentConverter
from docling_core.transforms.chunker.hybrid_chunker import HybridChunker
from docling_core.transforms.chunker.tokenizer.huggingface import HuggingFaceTokenizer
from docling_core.types.doc import DoclingDocument, SectionHeaderItem
from docling_core.types.doc.labels import DocItemLabel
from transformers import AutoTokenizer

from medibot.config import DENSE_MODEL, MAX_TOKENS, roles_for_collection

LEVEL_GAP_PT = 1.5  # heading heights closer than this (in PDF points) are treated as the same level
EMBED_WINDOW = 256  # all-MiniLM-L6-v2 truncates silently beyond this many tokens


@dataclass
class ChunkRecord:
    """One chunk, ready to embed and store. `text` is what gets embedded (breadcrumb + body)."""

    chunk_id: str
    text: str
    body: str
    source_document: str
    collection: str
    access_roles: list[str]
    section_title: str
    chunk_type: str
    heading_path: str
    chunk_index: int
    doc_hash: str
    n_tokens: int

    def payload(self) -> dict:
        """Qdrant payload: everything except the id (the id is the point id)."""
        d = asdict(self)
        d.pop("chunk_id")
        return d


_converter: DocumentConverter | None = None
_tokenizer = None
_chunker: HybridChunker | None = None


def converter() -> DocumentConverter:
    global _converter
    if _converter is None:
        _converter = DocumentConverter()
    return _converter


def chunker():
    """One tokenizer + chunker per process; the tokenizer is the embedding model's, so token caps match."""
    global _tokenizer, _chunker
    if _chunker is None:
        _tokenizer = AutoTokenizer.from_pretrained(DENSE_MODEL)
        _chunker = HybridChunker(
            tokenizer=HuggingFaceTokenizer(tokenizer=_tokenizer, max_tokens=MAX_TOKENS),
            merge_peers=True,
        )
    return _tokenizer, _chunker


def _height(item: SectionHeaderItem) -> float:
    bbox = item.prov[0].bbox
    return round(abs(bbox.t - bbox.b), 1)


def repair_heading_levels(doc: DoclingDocument) -> dict[int, int]:
    """Set SectionHeaderItem.level from bounding-box height clusters (largest = level 1).

    Returns {level: number_of_headings}. No-op for documents without layout boxes (markdown).
    """
    headers = [it for it, _ in doc.iterate_items() if isinstance(it, SectionHeaderItem) and it.prov]
    if not headers:
        return {}
    level_of: dict[float, int] = {}
    level, cluster_max = 0, None
    for h in sorted({_height(it) for it in headers}, reverse=True):
        if cluster_max is None or cluster_max - h > LEVEL_GAP_PT:
            level, cluster_max = level + 1, h
        level_of[h] = level
    counts: dict[int, int] = {}
    for it in headers:
        it.level = level_of[_height(it)]
        counts[it.level] = counts.get(it.level, 0) + 1
    return counts


def chunk_type_of(chunk) -> str:
    """Spec values: table | code | heading | text — decided by the Docling items inside the chunk."""
    labels = {it.label for it in chunk.meta.doc_items}
    if DocItemLabel.TABLE in labels:
        return "table"
    if DocItemLabel.CODE in labels:
        return "code"
    if labels and labels <= {DocItemLabel.SECTION_HEADER, DocItemLabel.TITLE}:
        return "heading"
    return "text"


def convert(path: Path) -> DoclingDocument:
    return converter().convert(str(path)).document


def chunk_document(doc: DoclingDocument, path: Path, collection: str) -> list[ChunkRecord]:
    """Repair heading levels, chunk, and attach the metadata schema required by the spec."""
    repair_heading_levels(doc)
    tokenizer, hc = chunker()
    doc_hash = hashlib.sha256(path.read_bytes()).hexdigest()[:16]
    access_roles = roles_for_collection(collection)
    records: list[ChunkRecord] = []
    for i, chunk in enumerate(hc.chunk(dl_doc=doc)):
        headings = [h for h in (chunk.meta.headings or []) if h]
        text = hc.contextualize(chunk=chunk)  # "H1\nH2\nH3\n<body>" — heading context in the embedded text
        records.append(
            ChunkRecord(
                chunk_id=str(uuid.uuid5(uuid.NAMESPACE_URL, f"medbud/{collection}/{path.name}/{i}")),
                text=text,
                body=chunk.text,
                source_document=path.name,
                collection=collection,
                access_roles=access_roles,
                section_title=headings[-1] if headings else path.stem,
                chunk_type=chunk_type_of(chunk),
                heading_path=" > ".join(headings) if headings else path.stem,
                chunk_index=i,
                doc_hash=doc_hash,
                n_tokens=len(tokenizer.encode(text)),
            )
        )
    return records


def parse_document(path: Path, collection: str) -> list[ChunkRecord]:
    return chunk_document(convert(path), path, collection)
