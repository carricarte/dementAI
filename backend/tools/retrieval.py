from __future__ import annotations

from typing import TypedDict

from backend.state.schema import Citation


class RetrievalResult(TypedDict):
    text: str
    citations: list[Citation]


def enrich_query(query: str, context: dict) -> str:
    """Append key clinical terms to a user query for better vector retrieval.

    Without enrichment the retriever matches on surface query words alone and
    returns tangential documents (e.g. a treatment-plan query retrieves PSP or
    NPH papers instead of AD-specific ones).  Adding the diagnosis, stage,
    scores, and APOE status shifts the embedding toward clinically relevant
    chunks.

    Args:
        query:   raw user query string
        context: flat dict with optional keys:
                 diagnosis, stage, mmse, cdr, apoe, medications (list[str])
    """
    parts = [query]
    if context.get("diagnosis"):
        parts.append(str(context["diagnosis"]))
    if context.get("stage"):
        parts.append(str(context["stage"]))
    if context.get("mmse") is not None:
        parts.append(f"MMSE {context['mmse']}")
    if context.get("cdr") is not None:
        parts.append(f"CDR {context['cdr']}")
    if context.get("apoe"):
        parts.append(str(context["apoe"]))
    if context.get("medications"):
        parts.append(", ".join(list(context["medications"])[:3]))
    return " ".join(parts)


def retrieve(
    query: str,
    source_filter: list[str] | None = None,
    top_k: int | None = None,
) -> RetrievalResult:
    from backend.rag.retriever import retriever  # late import avoids model load at startup

    text, citations = retriever.retrieve(query, source_filter=source_filter, top_k=top_k)
    return {"text": text, "citations": citations}
