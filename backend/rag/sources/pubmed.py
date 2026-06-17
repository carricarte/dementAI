"""PubMed E-utilities fetcher for the DCA knowledge base.

Reads search queries from backend/rag/pubmed_queries.toml, fetches matching
articles via NCBI E-utilities, and returns chunked Documents ready for LanceDB.

Rate limit: 3 requests/second without an API key. Set NCBI_API_KEY in .env to
raise this to 10/second (add it to backend/config.py if needed).

Requires optional [ingest] dependencies:
    pip install -e ".[ingest]"
"""

from __future__ import annotations

import time
import tomllib
import xml.etree.ElementTree as ET
from pathlib import Path

import httpx

from backend.rag.ingestion import Document, make_id

_BASE = "https://eutils.ncbi.nlm.nih.gov/entrez/eutils/"
# NCBI requires a contact email in the User-Agent for automated scripts.
_UA = "DCA-Research-Bot/1.0 (tcarricarte@gmail.com)"
_DELAY = 0.4  # seconds between requests — stays under 3 req/sec limit
_BATCH_SIZE = 100  # PMIDs per efetch call (NCBI max)

_DEFAULT_QUERIES_FILE = Path(__file__).parent.parent / "pubmed_queries.toml"

CHUNK_SIZE = 1000
CHUNK_OVERLAP = 150


# ── NCBI API helpers ────────────────────────────────────────────────────────


def _search(query: str, max_results: int, date_range_years: int) -> list[str]:
    """Return PMIDs matching query, sorted by relevance."""
    params: dict = {
        "db": "pubmed",
        "term": query,
        "retmax": max_results,
        "retmode": "json",
        "sort": "relevance",
    }
    if date_range_years:
        params["reldate"] = date_range_years * 365
        params["datetype"] = "pdat"

    with httpx.Client(timeout=30) as client:
        resp = client.get(f"{_BASE}esearch.fcgi", params=params, headers={"User-Agent": _UA})
        resp.raise_for_status()

    return resp.json().get("esearchresult", {}).get("idlist", [])


def _fetch_xml(pmids: list[str]) -> ET.Element:
    """Fetch PubMed XML for a batch of PMIDs."""
    params = {
        "db": "pubmed",
        "id": ",".join(pmids),
        "retmode": "xml",
        "rettype": "abstract",
    }
    with httpx.Client(timeout=60) as client:
        resp = client.get(f"{_BASE}efetch.fcgi", params=params, headers={"User-Agent": _UA})
        resp.raise_for_status()
    return ET.fromstring(resp.text)


# ── XML parsing ─────────────────────────────────────────────────────────────


def _parse_article(el: ET.Element) -> dict | None:
    """Parse a <PubmedArticle> element. Returns None if abstract is missing."""
    mc = el.find("MedlineCitation")
    if mc is None:
        return None

    pmid = mc.findtext("PMID", "").strip()
    art = mc.find("Article")
    if art is None:
        return None

    title = (art.findtext("ArticleTitle") or "").strip()

    # Abstract — structured abstracts have labelled <AbstractText> sections
    parts = []
    for ab in art.findall(".//AbstractText"):
        label = ab.get("Label", "")
        text = (ab.text or "").strip()
        if text:
            parts.append(f"{label}: {text}" if label else text)
    abstract = "\n".join(parts).strip()

    if not abstract:
        return None  # skip articles without accessible abstracts

    # Journal title
    journal = (
        art.findtext(".//Journal/Title") or art.findtext(".//Journal/ISOAbbreviation") or ""
    ).strip()

    # Publication year
    raw_year = (
        art.findtext(".//Journal/JournalIssue/PubDate/Year")
        or art.findtext(".//Journal/JournalIssue/PubDate/MedlineDate", "")[:4]
        or ""
    )
    year = int(raw_year) if raw_year.isdigit() else 0

    # Authors
    authors = []
    for author in art.findall(".//Author"):
        last = author.findtext("LastName", "")
        fore = author.findtext("ForeName", "")
        if last:
            authors.append(f"{last} {fore}".strip())

    # MeSH terms (only present after indexing — may be empty for new articles)
    mesh_terms = [
        mh.findtext("DescriptorName", "")
        for mh in mc.findall(".//MeshHeading")
        if mh.findtext("DescriptorName")
    ]

    # DOI
    doi = next(
        (a.text or "" for a in el.findall(".//ArticleId") if a.get("IdType") == "doi"),
        "",
    ).strip()

    return {
        "pmid": pmid,
        "doi": doi,
        "title": title,
        "abstract": abstract,
        "journal": journal,
        "year": year,
        "authors": authors,
        "mesh_terms": mesh_terms,
        "url": f"https://pubmed.ncbi.nlm.nih.gov/{pmid}/",
    }


# ── Chunking ────────────────────────────────────────────────────────────────


def _chunk(text: str, size: int = CHUNK_SIZE, overlap: int = CHUNK_OVERLAP) -> list[str]:
    if len(text) <= size:
        return [text]
    chunks: list[str] = []
    pos = 0
    while pos < len(text):
        end = min(pos + size, len(text))
        if end < len(text):
            for sep in ("\n\n", "\n", ". ", " "):
                cut = text.rfind(sep, pos + overlap, end)
                if cut != -1:
                    end = cut + len(sep)
                    break
        chunk = text[pos:end].strip()
        if chunk:
            chunks.append(chunk)
        pos = end - overlap if end < len(text) else end
    return chunks


# ── Public entry point ──────────────────────────────────────────────────────


def fetch(queries_file: Path | None = None) -> list[Document]:
    """Fetch PubMed articles for all queries defined in pubmed_queries.toml."""
    qfile = queries_file or _DEFAULT_QUERIES_FILE
    with open(qfile, "rb") as f:
        config = tomllib.load(f)

    cfg = config.get("settings", {})
    max_results: int = cfg.get("max_results_per_query", 100)
    date_range_years: int = cfg.get("date_range_years", 10)

    # Flatten queries preserving category label for progress output
    all_queries: list[tuple[str, str]] = [
        (category, q) for category, queries in config.get("queries", {}).items() for q in queries
    ]

    seen_pmids: set[str] = set()
    all_docs: list[Document] = []

    n_queries = len(all_queries)
    for q_idx, (category, query) in enumerate(all_queries, 1):
        print(f"\n  [{q_idx}/{n_queries}] [{category}] {query!r}", flush=True)
        print("    searching ...", end=" ", flush=True)
        try:
            pmids = _search(query, max_results, date_range_years)
            time.sleep(_DELAY)
        except Exception as e:
            print(f"search error: {e}", flush=True)
            continue

        new_pmids = [p for p in pmids if p not in seen_pmids]
        if not new_pmids:
            print(f"0 new PMIDs ({len(pmids)} hits, all seen)", flush=True)
            continue
        print(f"{len(new_pmids)} new / {len(pmids)} hits", flush=True)

        for i in range(0, len(new_pmids), _BATCH_SIZE):
            batch = new_pmids[i : i + _BATCH_SIZE]
            batch_num = i // _BATCH_SIZE + 1
            total_batches = (len(new_pmids) + _BATCH_SIZE - 1) // _BATCH_SIZE
            print(
                f"    batch {batch_num}/{total_batches} ({len(batch)} PMIDs) — downloading ...",
                end=" ",
                flush=True,
            )
            try:
                root = _fetch_xml(batch)
                time.sleep(_DELAY)
            except Exception as e:
                print(f"error: {e}", flush=True)
                continue

            articles = root.findall("PubmedArticle")
            print(f"got {len(articles)} articles — parsing ...", flush=True)

            for art_idx, article_el in enumerate(articles, 1):
                parsed = _parse_article(article_el)
                if not parsed or parsed["pmid"] in seen_pmids:
                    continue
                seen_pmids.add(parsed["pmid"])

                title_short = (
                    parsed["title"][:70] + "…" if len(parsed["title"]) > 70 else parsed["title"]
                )
                full_text = f"{parsed['title']}\n\n{parsed['abstract']}"
                chunks = _chunk(full_text)
                print(
                    f"      [{art_idx}/{len(articles)}] PMID {parsed['pmid']} "
                    f"→ {len(chunks)} chunk(s)  {title_short}",
                    flush=True,
                )
                for chunk_idx, chunk in enumerate(chunks):
                    all_docs.append(
                        Document(
                            id=make_id("pubmed", parsed["pmid"], chunk),
                            source="pubmed",
                            source_id=parsed["pmid"],
                            title=parsed["title"],
                            text=chunk,
                            url=parsed["url"],
                            pmid=parsed["pmid"],
                            doi=parsed["doi"],
                            journal=parsed["journal"],
                            authors=parsed["authors"],
                            mesh_terms=parsed["mesh_terms"],
                            year=parsed["year"],
                            chunk_index=chunk_idx,
                        )
                    )

    print(f"\n  [pubmed] total: {len(all_docs)} chunks from {len(seen_pmids)} articles", flush=True)
    return all_docs
