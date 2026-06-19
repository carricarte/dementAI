"""American Academy of Neurology (aan.com) fetcher.

Downloads publicly accessible clinical PDFs from aan.com, extracts and
chunks their text, and returns Documents ready for LanceDB ingestion.

Full AAN clinical practice guidelines are published in Neurology journal
(paywalled). The files below are freely accessible quality measurement sets
and a public-comment draft guideline.

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
_DEFAULT_DIR = Path("data/sources/aan")

# Freely downloadable files from aan.com relevant to dementia stages.
# Full CPGs are paywalled (published in Neurology journal); these are
# quality measurement sets and one public-comment draft guideline.
AAN_FILES: list[dict] = [
    {
        "url": "https://www.aan.com/siteassets/home-page/policy-and-guidelines/quality/quality-measures/2019.03.25-mci-measures.pdf",
        "filename": "aan-mci-quality-measures-2019.pdf",
        "source_id": "aan-mci-measures-2019",
        "title": "AAN Mild Cognitive Impairment Quality Measures",
        "year": 2019,
    },
    {
        "url": "https://www.aan.com/siteassets/home-page/policy-and-guidelines/quality/quality-measures/15dmmeasureset_pg.pdf",
        "filename": "aan-dementia-management-quality-set-2015.pdf",
        "source_id": "aan-dementia-mgmt-2015",
        "title": "Dementia Management Quality Measurement Set",
        "year": 2015,
    },
    {
        "url": "https://www.aan.com/globals/axon/assets/9493.pdf",
        "filename": "aan-dementia-performance-measures-2013.pdf",
        "source_id": "aan-dementia-perf-2013",
        "title": "Dementia Performance Measurement Set",
        "year": 2013,
    },
    {
        "url": "https://www.aan.com/siteassets/home-page/policy-and-guidelines/quality/quality-measures/2018-dementia-management-measures.pdf",
        "filename": "aan-dementia-management-measures-2018.pdf",
        "source_id": "aan-dementia-mgmt-2018",
        "title": "Dementia Management Quality Measurement Set (2018 Update)",
        "year": 2018,
    },
    {
        "url": "https://www.aan.com/siteassets/home-page/policy-and-guidelines/quality/quality-measures/movement-disorders/2020-parkinson-disease-measurement-set-final.pdf",
        "filename": "aan-parkinson-quality-measures-2020.pdf",
        "source_id": "aan-parkinson-2020",
        "title": "Parkinson's Disease Quality Measurement Set",
        "year": 2020,
    },
    {
        "url": "https://www.aan.com/siteassets/home-page/policy-and-guidelines/quality/quality-measures/neuromuscular/23-als-measure-update-final.pdf",
        "filename": "aan-als-quality-measures-2023.pdf",
        "source_id": "aan-als-2023",
        "title": "Amyotrophic Lateral Sclerosis (ALS) Quality Measures",
        "year": 2023,
    },
]
# Excluded (public comment draft, not formally published):
# "Etiologic Diagnosis of Dementia" (2018) — aan.com/.../18dementiadiagnosisprotocolforpubcom_pg.pdf
# Final published version is in Neurology journal (paywalled).


def _download_file(url: str, dest: Path) -> Path:
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
    """Download AAN clinical files and return chunked Documents."""
    root = sources_dir or _DEFAULT_DIR
    root.mkdir(parents=True, exist_ok=True)

    all_docs: list[Document] = []

    for entry in AAN_FILES:
        print(f"\n  {entry['title'][:80]} ...")
        dest = root / entry["filename"]

        try:
            file_path = _download_file(entry["url"], dest)
            time.sleep(0.5)
        except Exception as e:
            print(f"  [error] Download failed for {entry['filename']}: {e}")
            continue

        print(f"  Parsing {file_path.name} ...")
        try:
            pages = _extract_pages(file_path)
        except Exception as e:
            print(f"  [error] PDF parse failed: {e}")
            continue

        chunks = _chunk_pages(pages)
        print(f"  → {len(chunks)} chunks across {len(pages)} pages")

        for page_num, chunk_idx, text in chunks:
            all_docs.append(
                Document(
                    id=make_id("aan", entry["source_id"], text),
                    source="aan",
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
