"""
Ingest documents into the DementIA knowledge base (LanceDB).

Usage:
    python scripts/ingest.py --source awmf
    python scripts/ingest.py --source pubmed
    python scripts/ingest.py --all
    python scripts/ingest.py --source pubmed --drop   # drop & recreate table first

Requires the [ingest,rag] optional dependencies:
    pip install -e ".[ingest,rag]"
"""

from __future__ import annotations

import argparse
import sys

sys.path.insert(0, ".")

from backend.rag.ingestion import Document, ingest

SOURCES = ["pubmed", "clinicaltrials", "neurology", "alz", "aan", "awmf"]


def fetch_awmf() -> list[Document]:
    from backend.rag.sources.awmf import fetch

    return fetch()


def fetch_pubmed() -> list[Document]:
    from backend.rag.sources.pubmed import fetch

    return fetch()


def fetch_clinicaltrials() -> list[Document]:
    raise NotImplementedError("ClinicalTrials fetcher not yet implemented")


def fetch_neurology() -> list[Document]:
    raise NotImplementedError("Neurology.org fetcher not yet implemented")


def fetch_alz() -> list[Document]:
    from backend.rag.sources.alz import fetch

    return fetch()


def fetch_aan() -> list[Document]:
    from backend.rag.sources.aan import fetch

    return fetch()


def main() -> None:
    parser = argparse.ArgumentParser(description="Populate the DementIA LanceDB knowledge base")
    parser.add_argument("--source", choices=SOURCES, help="Single source to ingest")
    parser.add_argument("--all", action="store_true", help="Ingest all sources")
    parser.add_argument(
        "--drop",
        action="store_true",
        help="Drop and recreate the LanceDB table before ingesting (use after schema changes)",
    )
    args = parser.parse_args()

    if not args.all and not args.source:
        parser.print_help()
        return

    sources = SOURCES if args.all else [args.source]
    total = 0
    drop_on_first = args.drop  # only drop once, on the first source

    for src in sources:
        print(f"\n[{src}] Fetching ...", flush=True)
        try:
            if src == "awmf":
                docs = fetch_awmf()
            elif src == "pubmed":
                docs = fetch_pubmed()
            elif src == "clinicaltrials":
                docs = fetch_clinicaltrials()
            elif src == "neurology":
                docs = fetch_neurology()
            elif src == "alz":
                docs = fetch_alz()
            elif src == "aan":
                docs = fetch_aan()
            else:
                print(f"  {src}: no fetcher, skipping", flush=True)
                continue

            if not docs:
                print(f"  {src}: 0 documents returned", flush=True)
                continue

            print(f"  {len(docs)} chunks fetched — embedding ...", flush=True)
            n = ingest(docs, drop_existing=drop_on_first)
            drop_on_first = False  # only drop on the first successful ingest
            print(f"  {src}: ingested {n} chunks", flush=True)
            total += n

        except NotImplementedError as e:
            print(f"  {src}: {e}", flush=True)
        except Exception as e:
            print(f"  {src}: unexpected error — {e}", flush=True)
            raise

    print(f"\nDone. Total chunks ingested: {total}", flush=True)


if __name__ == "__main__":
    main()
