"""AWMF Leitlinien fetcher.

Downloads the S3-Leitlinie Demenzen (038-013) and related guidelines from
the AWMF Leitlinien-Register REST API, extracts and chunks their text, and
returns Documents ready for LanceDB ingestion.

The API endpoint and its public API key are embedded in the AWMF register
frontend JavaScript — they are intentionally public (client-side) credentials.

Requires optional [ingest] + [rag] dependencies:
    pip install -e ".[ingest,rag]"
"""

from __future__ import annotations

import re
import time
from pathlib import Path

import httpx

from backend.rag.chunker import chunker
from backend.rag.ingestion import Document, make_id

# Register number → canonical title
AWMF_TARGETS: dict[str, str] = {
    "038-013": "S3-Leitlinie Demenzen",
}

# AWMF Leitlinien-Register REST API (public key embedded in register.awmf.org frontend)
_API_BASE = "https://leitlinien-api.awmf.org/v1/"
_API_KEY = "MkI5Y1VIOEJ0ZGpoelNBVXRNM1E6WVFld0pBUF9RLVdJa012UHVPTmRQUQ=="
_ASSETS_BASE = "https://register.awmf.org/assets/guidelines/"
_UA = "DCA-Research-Bot/1.0 (academic dementia research)"
_DEFAULT_DIR = Path("data/sources/awmf")

# Document types to ingest from the API links list (in preference order)
_INGEST_TYPES = {"longVersion"}


# ---------------------------------------------------------------------------
# API helpers
# ---------------------------------------------------------------------------


def _api_headers() -> dict:
    return {"User-Agent": _UA, "Api-Key": _API_KEY, "Accept": "application/json"}


def _fetch_guideline_meta(register_num: str) -> dict:
    """Return the first active record from the AWMF API for a register number."""
    assoc, num = register_num.split("-", 1)
    url = f"{_API_BASE}get/{assoc}/{num}?limit=20&lang=de"
    with httpx.Client(timeout=30, follow_redirects=True) as client:
        resp = client.get(url, headers=_api_headers())
        resp.raise_for_status()
    records = resp.json().get("records", [])
    if not records:
        raise RuntimeError(f"AWMF API returned no records for {register_num}")
    # First record is the most recent (sorted by releaseDate:desc)
    return records[0]


def _select_pdfs(meta: dict, ingest_types: set[str]) -> list[dict]:
    """Return links that have an active PDF of an ingestable type."""
    results = []
    for link in meta.get("links", []):
        if link.get("type") not in ingest_types:
            continue
        media = link.get("media", "")
        if not media or not link.get("active"):
            continue
        url = _ASSETS_BASE + media
        results.append({"url": url, "filename": media, "type": link["type"]})
    return results


# ---------------------------------------------------------------------------
# Download helpers
# ---------------------------------------------------------------------------


def _download_pdf(url: str, dest: Path) -> Path:
    if dest.exists():
        print(f"    [cached]   {dest.name}")
        return dest
    dest.parent.mkdir(parents=True, exist_ok=True)
    print(f"    [download] {url}")
    with httpx.Client(timeout=120, follow_redirects=True) as client:
        resp = client.get(url, headers={"User-Agent": _UA})
        resp.raise_for_status()
    dest.write_bytes(resp.content)
    print(f"    [saved]    {dest.name} ({len(resp.content) // 1024} KB)")
    return dest


# ---------------------------------------------------------------------------
# PDF parsing and chunking
# ---------------------------------------------------------------------------


def _extract_pages(pdf_path: Path) -> list[tuple[int, str]]:
    """Return (page_number, text) for each content-bearing page.

    Preserves sentence-ending newlines so the chunker can split on them.
    Only collapses horizontal whitespace (spaces/tabs) and mid-word hyphens.
    """
    from pypdf import PdfReader  # optional [ingest] dep

    reader = PdfReader(str(pdf_path))
    pages = []
    for i, page in enumerate(reader.pages, start=1):
        raw = page.extract_text() or ""
        text = re.sub(r"-\n", "", raw)  # dehyphenate line-end hyphens
        text = re.sub(r"[ \t]+", " ", text)  # collapse spaces/tabs only
        text = re.sub(r"\n{3,}", "\n\n", text)  # normalise excessive blank lines
        text = text.strip()
        if len(text) > 80:
            pages.append((i, text))
    return pages


def _chunk_pages(pages: list[tuple[int, str]]) -> list[tuple[int, int, str]]:
    """Chunk all pages with the token-aware sentence chunker.

    Returns (page_number, chunk_index, text); chunk_index is global across pages.
    """
    result: list[tuple[int, int, str]] = []
    chunk_idx = 0
    for page_num, text in pages:
        for chunk_text in chunker.chunk(text):
            result.append((page_num, chunk_idx, chunk_text))
            chunk_idx += 1
    return result


def _year_from_filename(name: str) -> int:
    m = re.search(r"(20\d{2})", name)
    return int(m.group(1)) if m else 0


# ---------------------------------------------------------------------------
# Public entry point
# ---------------------------------------------------------------------------


def fetch(sources_dir: Path | None = None) -> list[Document]:
    """Download AWMF guidelines via the REST API and return chunked Documents."""
    root = sources_dir or _DEFAULT_DIR
    root.mkdir(parents=True, exist_ok=True)

    all_docs: list[Document] = []

    for register_num, default_title in AWMF_TARGETS.items():
        print(f"\n  AWMF {register_num}")
        guide_dir = root / register_num
        guide_dir.mkdir(parents=True, exist_ok=True)

        # 1. Fetch guideline metadata from the AWMF REST API
        try:
            meta = _fetch_guideline_meta(register_num)
        except Exception as e:
            print(f"  [error] API call failed: {e}")
            continue

        title = meta.get("name", default_title)
        release_date = meta.get("releaseDate", "")
        year = _year_from_filename(release_date) or _year_from_filename(meta.get("lastEdit", ""))
        print(f"  {title} — v{meta.get('version', '?')} ({release_date})")

        # 2. Select PDFs to ingest
        pdf_links = _select_pdfs(meta, _INGEST_TYPES)
        if not pdf_links:
            print("  [warn] No active PDFs found via API")
            continue
        print(f"  PDFs to ingest: {[lnk['filename'] for lnk in pdf_links]}")

        # 3. Download and process each PDF
        for link in pdf_links:
            dest = guide_dir / link["filename"]
            try:
                pdf_path = _download_pdf(link["url"], dest)
                time.sleep(0.5)  # polite rate-limiting
            except Exception as e:
                print(f"  [error] Download failed for {link['filename']}: {e}")
                continue

            print(f"  Parsing {pdf_path.name} ({link['type']}) ...")
            try:
                pages = _extract_pages(pdf_path)
            except Exception as e:
                print(f"  [error] PDF parse failed: {e}")
                continue

            chunks = _chunk_pages(pages)
            print(f"  → {len(chunks)} chunks across {len(pages)} pages")

            for page_num, chunk_idx, text in chunks:
                all_docs.append(
                    Document(
                        id=make_id("awmf", register_num, text),
                        source="awmf",
                        source_id=register_num,
                        title=f"{title} ({link['type']})",
                        text=text,
                        url=link["url"],
                        year=year,
                        page=page_num,
                        chunk_index=chunk_idx,
                    )
                )

    return all_docs
