"""Unit tests for the aan.com knowledge source fetcher."""

from __future__ import annotations

from pathlib import Path

import httpx
import pytest

from backend.rag.ingestion import Document

_CACHE_DIR = Path("data/sources/aan")
_has_cache = (_CACHE_DIR / "aan-mci-quality-measures-2019.pdf").exists()
requires_cache = pytest.mark.skipif(not _has_cache, reason="Cached AAN PDFs not available")


@requires_cache
def test_fetch_returns_documents():
    from backend.rag.sources.aan import fetch

    docs = fetch(sources_dir=_CACHE_DIR)

    assert len(docs) > 0
    assert all(isinstance(d, Document) for d in docs)


@requires_cache
def test_fetch_document_source():
    from backend.rag.sources.aan import fetch

    docs = fetch(sources_dir=_CACHE_DIR)

    assert all(d.source == "aan" for d in docs)


@requires_cache
def test_fetch_all_files_represented():
    from backend.rag.sources.aan import fetch

    docs = fetch(sources_dir=_CACHE_DIR)
    source_ids = {d.source_id for d in docs}

    assert "aan-mci-measures-2019" in source_ids
    assert "aan-dementia-mgmt-2015" in source_ids
    assert "aan-dementia-perf-2013" in source_ids


@requires_cache
def test_fetch_document_fields_populated():
    from backend.rag.sources.aan import fetch

    docs = fetch(sources_dir=_CACHE_DIR)

    for doc in docs:
        assert doc.id
        assert doc.text
        assert doc.title
        assert doc.url.startswith("https://www.aan.com/")
        assert doc.year > 0
        assert doc.page >= 1
        assert doc.chunk_index >= 0


@requires_cache
def test_fetch_document_ids_unique():
    from backend.rag.sources.aan import fetch

    docs = fetch(sources_dir=_CACHE_DIR)
    ids = [d.id for d in docs]

    assert len(ids) == len(set(ids))


def test_fetch_skips_failed_download(tmp_path, monkeypatch):
    from backend.rag.sources import aan as aan_mod

    def _fail(url: str, dest: Path) -> Path:
        raise httpx.HTTPError("connection refused")

    monkeypatch.setattr(aan_mod, "_download_file", _fail)
    docs = aan_mod.fetch(sources_dir=tmp_path)

    assert docs == []


def test_fetch_skips_unparseable_pdf(tmp_path, monkeypatch):
    from backend.rag.sources import aan as aan_mod

    corrupt = tmp_path / "aan-mci-quality-measures-2019.pdf"
    corrupt.write_bytes(b"not a real pdf")

    def _return_cached(url: str, dest: Path) -> Path:
        return corrupt

    monkeypatch.setattr(aan_mod, "_download_file", _return_cached)
    docs = aan_mod.fetch(sources_dir=tmp_path)

    assert docs == []
