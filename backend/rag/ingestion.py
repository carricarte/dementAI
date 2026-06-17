from __future__ import annotations

import hashlib
from dataclasses import dataclass, field

import lancedb
import pyarrow as pa

from backend.config import settings
from backend.rag.embedder import embedder
from backend.rag.retriever import TABLE_NAME


def make_id(source: str, source_id: str, text: str) -> str:
    """Stable content-hash ID: same source + document + chunk text → same ID.

    Using source_id (PMID, register number, …) in the hash means two sources
    that happen to share identical text still get distinct IDs.
    """
    digest = hashlib.sha256(f"{source}\x00{source_id}\x00{text}".encode()).hexdigest()
    return f"{source}_{digest[:20]}"


VECTOR_DIM = 768  # PubMedBERT / neuml/pubmedbert-base-embeddings output dim

# ── Schema ─────────────────────────────────────────────────────────────────
# Common fields (all sources): id, source, source_id, title, text, url,
#                              authors, year, chunk_index, vector
# PubMed-specific:             pmid, doi, journal, mesh_terms
# AWMF-specific:               page (PDF page number)
# Fields absent for a given source are stored as empty string / empty list / 0.

SCHEMA = pa.schema(
    [
        # ── identity ──────────────────────────────────────────────────────────
        pa.field("id", pa.string()),
        pa.field("source", pa.string()),  # awmf | pubmed | clinicaltrials | …
        pa.field("source_id", pa.string()),  # register number, PMID, NCT ID, …
        # ── content ───────────────────────────────────────────────────────────
        pa.field("title", pa.string()),
        pa.field("text", pa.string()),  # indexed chunk
        # ── common metadata ───────────────────────────────────────────────────
        pa.field("url", pa.string()),
        pa.field("authors", pa.list_(pa.string())),
        pa.field("year", pa.int32()),
        pa.field("chunk_index", pa.int32()),
        # ── PubMed-specific ───────────────────────────────────────────────────
        pa.field("pmid", pa.string()),
        pa.field("doi", pa.string()),
        pa.field("journal", pa.string()),
        pa.field("mesh_terms", pa.list_(pa.string())),
        # ── AWMF-specific ─────────────────────────────────────────────────────
        pa.field("page", pa.int32()),  # PDF page number
        # ── vector ────────────────────────────────────────────────────────────
        pa.field("vector", pa.list_(pa.float32(), VECTOR_DIM)),
    ]
)


@dataclass
class Document:
    # required
    id: str
    source: str  # "awmf", "pubmed", etc.
    title: str
    text: str  # the indexed chunk

    # common optional
    source_id: str = ""  # register number, PMID, NCT number, …
    url: str = ""
    authors: list[str] = field(default_factory=list)
    year: int = 0
    chunk_index: int = 0

    # PubMed-specific
    pmid: str = ""
    doi: str = ""
    journal: str = ""
    mesh_terms: list[str] = field(default_factory=list)

    # AWMF-specific
    page: int = 0  # PDF page number; 0 = N/A


def ingest(documents: list[Document], drop_existing: bool = False) -> int:
    db = lancedb.connect(settings.lancedb_path)

    if drop_existing:
        try:
            db.drop_table(TABLE_NAME)
        except Exception:
            pass

    try:
        tbl = db.open_table(TABLE_NAME)
    except Exception:
        tbl = db.create_table(TABLE_NAME, schema=SCHEMA)

    # Skip documents whose IDs are already in the table.
    try:
        existing_ids: set[str] = set(
            tbl.to_lance().to_table(columns=["id"]).column("id").to_pylist()
        )
    except Exception:
        existing_ids = set()

    new_docs = [d for d in documents if d.id not in existing_ids]
    if not new_docs:
        return 0

    rows = []
    for doc in new_docs:
        vector = embedder.embed_article(doc.title + " " + doc.text)
        rows.append(
            {
                "id": doc.id,
                "source": doc.source,
                "source_id": doc.source_id,
                "title": doc.title,
                "text": doc.text,
                "url": doc.url,
                "authors": doc.authors,
                "year": doc.year,
                "chunk_index": doc.chunk_index,
                "pmid": doc.pmid,
                "doi": doc.doi,
                "journal": doc.journal,
                "mesh_terms": doc.mesh_terms,
                "page": doc.page,
                "vector": vector,
            }
        )

    tbl.add(rows)
    return len(rows)
