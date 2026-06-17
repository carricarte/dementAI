from __future__ import annotations

import lancedb

from backend.config import settings
from backend.rag.embedder import embedder
from backend.state.schema import Citation

TABLE_NAME = "knowledge"

_EXPAND_PROMPT = """\
You are a clinical information retrieval assistant.

Given a clinical question or patient presentation, produce exactly 3 concise search queries \
to retrieve relevant evidence from a biomedical literature database. Each query must:
- Be 5–15 words using precise medical terminology
- Target a distinct aspect: e.g. diagnostic criteria, differential workup, biomarkers, treatment

Return only the 3 queries, one per line, no numbering or other text.

Clinical question: {query}"""


def _expand_query(query: str) -> list[str]:
    """Use the LLM to decompose a clinical narrative into focused retrieval sub-queries."""
    from backend.llm import get_llm  # late import — model already loaded by caller

    response = get_llm().invoke(_EXPAND_PROMPT.format(query=query))
    lines = [line.strip() for line in response.content.strip().splitlines() if line.strip()]
    return lines[:3]


class KnowledgeRetriever:
    def __init__(self) -> None:
        self._db = lancedb.connect(settings.lancedb_path)

    def retrieve(
        self,
        query: str,
        source_filter: list[str] | None = None,
        top_k: int | None = None,
        expand: bool = True,
    ) -> tuple[str, list[Citation]]:
        k = top_k or settings.retrieval_top_k

        try:
            tbl = self._db.open_table(TABLE_NAME)
        except Exception:
            return "", []  # knowledge base not yet populated

        # Build list of queries: original + up to 3 LLM-generated sub-queries.
        queries = [query]
        if expand:
            try:
                queries += _expand_query(query)
            except Exception:
                pass  # fall back to single query if LLM call fails

        # Run vector search for each query, tracking the BEST (lowest) distance
        # each chunk achieves across all queries. This ensures a chunk surfaced by
        # any sub-query is ranked on its strongest match, not its first appearance.
        fetch_limit = k * 4  # fetch generously before source filtering
        best: dict[str, tuple[float, object]] = {}  # id → (best_distance, row)

        for q in queries:
            vector = embedder.embed_query(q)
            results = tbl.search(vector).limit(fetch_limit).to_pandas()
            if source_filter:
                results = results[results["source"].isin(source_filter)]
            for _, row in results.iterrows():
                dist = float(row.get("_distance", 1.0))
                rid = row["id"]
                if rid not in best or dist < best[rid][0]:
                    best[rid] = (dist, row)

        # Sort globally by best distance, take top-k.
        top_rows = [row for _, row in sorted(best.values(), key=lambda x: x[0])][:k]

        chunks = [row["text"] for row in top_rows]

        # Deduplicate citations by (source, title), preserving retrieval order.
        seen_cite: set[tuple[str, str]] = set()
        citations: list[Citation] = []
        for row in top_rows:
            key = (row["source"], row["title"])
            if key in seen_cite:
                continue
            seen_cite.add(key)
            citations.append(
                Citation(
                    source=row["source"],
                    title=row["title"],
                    url=row.get("url") or None,
                    pmid=row.get("pmid") or None,
                )
            )

        return "\n\n---\n\n".join(chunks), citations


retriever = KnowledgeRetriever()
