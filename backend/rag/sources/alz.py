"""Alzheimer's Association (alz.org) fetcher.

Downloads publicly accessible clinical PDFs from alz.org, extracts and
chunks their text, and returns Documents ready for LanceDB ingestion.

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

_UA = "DementIA-Research-Bot/1.0 (academic dementia research)"
_DEFAULT_DIR = Path("data/sources/alz")

# Publicly accessible clinical PDFs from alz.org relevant to the
# dementia knowledge base stages (screening, diagnosis, treatment, care).
# pro.alz.org content is JS-rendered and inaccessible without a browser.
ALZ_PDFS: list[dict] = [
    {
        "url": "https://www.alz.org/getmedia/fb6a8416-8d51-4132-b41c-96caf39f9e6a/jalz-1528.pdf",
        "filename": "jalz-1528.pdf",
        "source_id": "jalz-1528",
        "title": (
            "Alzheimer's Association recommendations for operationalizing the detection "
            "of cognitive impairment during the Medicare Annual Wellness Visit in a "
            "primary care setting"
        ),
        "year": 2013,
    },
    {
        "url": "https://www.alz.org/getmedia/31b11f61-4e72-4353-91bb-3e12825553bd/lecanemab-toolkit.pdf",
        "filename": "lecanemab-toolkit.pdf",
        "source_id": "lecanemab-aur",
        "title": "Lecanemab Appropriate Use Recommendations and Clinical Toolkit",
        "year": 2023,
    },
    # ── Screening tool PDFs ──────────────────────────────────────────────────
    {
        "url": "https://www.alz.org/getmedia/7c51d895-9df8-4819-8ea3-b896ff9a9deb/mini-cog.pdf",
        "filename": "mini-cog.pdf",
        "source_id": "mini-cog",
        "title": "Mini-Cog: A 3-minute cognitive screen for dementia",
        "year": 2023,
    },
    {
        "url": "https://www.alz.org/getmedia/a195e9f7-5dea-407a-83a9-2bc80e4c4796/gpcog-screening-test-english.pdf",
        "filename": "gpcog-screening-test-english.pdf",
        "source_id": "gpcog",
        "title": "GPCOG: General Practitioner Assessment of Cognition",
        "year": 2023,
    },
    {
        "url": "https://www.alz.org/getmedia/2cc07bd9-1299-42c3-ac54-a61388ee4ef1/memory-impairment-screening-mis.pdf",
        "filename": "memory-impairment-screening-mis.pdf",
        "source_id": "mis",
        "title": "Memory Impairment Screen (MIS)",
        "year": 2023,
    },
    {
        "url": "https://www.alz.org/getmedia/6e7291bf-4ac8-40ed-a148-824d4591ed7e/ad8-dementia-screening.pdf",
        "filename": "ad8-dementia-screening.pdf",
        "source_id": "ad8",
        "title": "AD8: Eight-Item Informant Interview to Differentiate Aging and Dementia",
        "year": 2023,
    },
    {
        "url": "https://www.alz.org/getmedia/77436b38-a073-4eca-8298-46552ab94c17/short-form-informant-questionnaire-decline.pdf",
        "filename": "short-form-informant-questionnaire-decline.pdf",
        "source_id": "iqcode-sf",
        "title": "Short-Form Informant Questionnaire on Cognitive Decline in the Elderly (IQCODE)",
        "year": 2023,
    },
    # ── Epidemiology and care statistics ────────────────────────────────────
    {
        "url": "https://www.alz.org/getmedia/a6b0adee-d708-43a0-b78d-44bdd9234b60/alzheimers-facts-and-figures-special-report-2025.pdf",
        "filename": "alzheimers-facts-and-figures-special-report-2025.pdf",
        "source_id": "alz-facts-figures-2025-sr",
        "title": "2025 Alzheimer's Disease Facts and Figures Special Report",
        "year": 2025,
    },
]


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


def _extract_pages(pdf_path: Path) -> list[tuple[int, str]]:
    from pypdf import PdfReader

    reader = PdfReader(str(pdf_path))
    pages = []
    for i, page in enumerate(reader.pages, start=1):
        raw = page.extract_text() or ""
        text = re.sub(r"-\n", "", raw)
        text = re.sub(r"[ \t]+", " ", text)
        text = re.sub(r"\n{3,}", "\n\n", text)
        text = text.strip()
        if len(text) > 80:
            pages.append((i, text))
    return pages


def _chunk_pages(pages: list[tuple[int, str]]) -> list[tuple[int, int, str]]:
    result: list[tuple[int, int, str]] = []
    chunk_idx = 0
    for page_num, text in pages:
        for chunk_text in chunker.chunk(text):
            result.append((page_num, chunk_idx, chunk_text))
            chunk_idx += 1
    return result


def fetch(sources_dir: Path | None = None) -> list[Document]:
    """Download alz.org clinical PDFs and return chunked Documents."""
    root = sources_dir or _DEFAULT_DIR
    root.mkdir(parents=True, exist_ok=True)

    all_docs: list[Document] = []

    for entry in ALZ_PDFS:
        print(f"\n  {entry['title'][:80]} ...")
        dest = root / entry["filename"]

        try:
            pdf_path = _download_pdf(entry["url"], dest)
            time.sleep(0.5)
        except Exception as e:
            print(f"  [error] Download failed for {entry['filename']}: {e}")
            continue

        print(f"  Parsing {pdf_path.name} ...")
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
                    id=make_id("alz", entry["source_id"], text),
                    source="alz",
                    source_id=entry["source_id"],
                    title=entry["title"],
                    text=text,
                    url=entry["url"],
                    year=entry["year"],
                    page=page_num,
                    chunk_index=chunk_idx,
                )
            )

    return all_docs
