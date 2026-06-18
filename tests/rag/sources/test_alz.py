"""Unit tests for the alz.org knowledge source fetcher."""

from __future__ import annotations

from pathlib import Path

import httpx
import pytest

from backend.rag.ingestion import Document

_CACHE_DIR = Path("data/sources/alz")
_has_cache = (_CACHE_DIR / "jalz-1528.pdf").exists()
requires_cache = pytest.mark.skipif(not _has_cache, reason="Cached alz.org PDFs not available")


@requires_cache
def test_fetch_returns_documents():
    """fetch() produces non-empty list of Documents from cached PDFs."""
    from backend.rag.sources.alz import fetch

    docs = fetch(sources_dir=_CACHE_DIR)

    assert len(docs) > 0
    assert all(isinstance(d, Document) for d in docs)


@requires_cache
def test_fetch_document_source():
    """All documents have source='alz'."""
    from backend.rag.sources.alz import fetch

    docs = fetch(sources_dir=_CACHE_DIR)

    assert all(d.source == "alz" for d in docs)


@requires_cache
def test_fetch_both_pdfs_represented():
    """Both configured PDFs produce at least one chunk each."""
    from backend.rag.sources.alz import fetch

    docs = fetch(sources_dir=_CACHE_DIR)
    source_ids = {d.source_id for d in docs}

    assert "jalz-1528" in source_ids
    assert "lecanemab-aur" in source_ids


@requires_cache
def test_fetch_document_fields_populated():
    """Every document has id, text, title, url, year, page, and chunk_index set."""
    from backend.rag.sources.alz import fetch

    docs = fetch(sources_dir=_CACHE_DIR)

    for doc in docs:
        assert doc.id
        assert doc.text
        assert doc.title
        assert doc.url.startswith("https://www.alz.org/")
        assert doc.year > 0
        assert doc.page >= 1
        assert doc.chunk_index >= 0


@requires_cache
def test_fetch_document_ids_unique():
    """No two documents share the same id (dedup key)."""
    from backend.rag.sources.alz import fetch

    docs = fetch(sources_dir=_CACHE_DIR)
    ids = [d.id for d in docs]

    assert len(ids) == len(set(ids))


def test_fetch_skips_failed_download(tmp_path, monkeypatch):
    """fetch() skips a PDF that fails to download and returns empty list."""
    from backend.rag.sources import alz as alz_mod

    def _fail(url: str, dest: Path) -> Path:
        raise httpx.HTTPError("connection refused")

    monkeypatch.setattr(alz_mod, "_download_pdf", _fail)
    docs = alz_mod.fetch(sources_dir=tmp_path)

    assert docs == []


def test_fetch_skips_unparseable_pdf(tmp_path, monkeypatch):
    """fetch() skips a PDF that fails to parse and continues."""
    from backend.rag.sources import alz as alz_mod

    # Write a corrupt file so _download_pdf "succeeds" but parse fails
    corrupt = tmp_path / "jalz-1528.pdf"
    corrupt.write_bytes(b"not a real pdf")

    def _return_cached(url: str, dest: Path) -> Path:
        return corrupt

    monkeypatch.setattr(alz_mod, "_download_pdf", _return_cached)
    docs = alz_mod.fetch(sources_dir=tmp_path)

    assert docs == []
