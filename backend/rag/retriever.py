from __future__ import annotations

from collections import defaultdict
from functools import lru_cache
from typing import Literal

import lancedb

from backend.config import settings
from backend.rag.embedder import embedder
from backend.state.schema import Citation

TABLE_NAME = "knowledge"

SearchStrategy = Literal["vector", "fts", "hybrid", "hybrid_rerank"]

_EXPAND_PROMPT = """\
You are a biomedical information retrieval assistant.

Given a clinical question, produce exactly 3 search queries to retrieve relevant evidence \
from a biomedical literature database. Each query must:
- Preserve the original intent exactly — if the question asks about causes, all 3 queries \
must be about causes; if about treatment, all 3 about treatment. Do NOT pivot to other \
clinical aspects.
- Use different but equivalent terminology: synonyms, MeSH terms, gene names, pathological \
terms, abbreviations — to improve recall across different phrasings in the literature.
- Be 5–15 words of precise medical terminology.

Return only the 3 queries, one per line, no numbering or other text.

Clinical question: {query}"""


@lru_cache(maxsize=256)
def _expand_query(query: str) -> list[str]:
    """Decompose a clinical narrative into 3 focused retrieval sub-queries via LLM."""
    from backend.llm import get_llm  # late import — model already loaded by caller

    llm = get_llm()
    response = llm.invoke(_EXPAND_PROMPT.format(query=query))
    lines = [line.strip() for line in response.content.strip().splitlines() if line.strip()]
    return lines[:3]


class _LazyReranker:
    """Lazy-loaded cross-encoder; reloads only if model_name changes."""

    _model = None
    _model_name: str | None = None

    def rerank(self, query: str, rows: list, top_k: int, model_name: str) -> list:
        try:
            if self._model is None or self._model_name != model_name:
                from sentence_transformers import CrossEncoder

                self._model = CrossEncoder(model_name, trust_remote_code=True)
                self._model_name = model_name
            pairs = [(query, row["text"]) for row in rows]
            scores = self._model.predict(pairs)
            ranked = sorted(zip(scores, rows), key=lambda x: x[0], reverse=True)
            return [row for _, row in ranked[:top_k]]
        except Exception:
            # sentence-transformers not installed or model download failed
            return rows[:top_k]


_reranker = _LazyReranker()


class KnowledgeRetriever:
    def __init__(self) -> None:
        self._db = lancedb.connect(settings.lancedb_path)

    def retrieve(
        self,
        query: str,
        source_filter: list[str] | None = None,
        top_k: int | None = None,
        expand: bool = True,
        strategy: SearchStrategy | None = None,
    ) -> tuple[str, list[Citation]]:
        k = top_k or settings.retrieval_top_k
        strat = strategy or settings.search_strategy  # type: ignore[assignment]

        try:
            tbl = self._db.open_table(TABLE_NAME)
        except Exception:
            return "", []

        # Multi-query expansion: original query + up to 3 focused sub-queries.
        queries = [query]
        if expand:
            try:
                queries += _expand_query(query)
            except Exception:
                pass

        fetch_limit = k * 4  # fetch generously; source filter may reduce count

        if strat == "vector":
            candidates = self._vector(tbl, queries, source_filter, fetch_limit)
        elif strat == "fts":
            candidates = self._fts(tbl, queries, source_filter, fetch_limit)
        elif strat in ("hybrid", "hybrid_rerank"):
            candidates = self._hybrid(tbl, queries, source_filter, fetch_limit)
        else:
            candidates = self._vector(tbl, queries, source_filter, fetch_limit)

        if strat == "hybrid_rerank":
            # Cross-encoder rerank a large candidate pool against the original
            # query — sub-queries expand recall, reranker refines by intent.
            candidates = _reranker.rerank(
                query,
                candidates[: k * 10],
                k,
                settings.rerank_model,
            )
        else:
            candidates = candidates[:k]

        def _clean(v: object) -> str | None:
            return str(v) if v and str(v) not in ("None", "nan") else None

        # Assign each unique (source, title) a citation number in ranked order.
        cite_num: dict[tuple[str, str], int] = {}
        citations: list[Citation] = []
        for row in candidates:
            key = (row["source"], row["title"])
            if key not in cite_num:
                n = len(cite_num) + 1
                cite_num[key] = n
                citations.append(
                    Citation(
                        source=row["source"],
                        title=row["title"],
                        url=_clean(row.get("url")),
                        pmid=_clean(row.get("pmid")),
                    )
                )

        # Prefix each chunk with its citation number so the LLM can inline-cite.
        numbered = [
            f"[{cite_num[(row['source'], row['title'])]}] {row['text']}" for row in candidates
        ]

        return "\n\n---\n\n".join(numbered), citations

    # ── private search helpers ─────────────────────────────────────────────

    def _vector(self, tbl, queries, source_filter, fetch_limit) -> list:
        """Multi-query vector search; merge by best distance per chunk."""
        best: dict[str, tuple[float, object]] = {}
        for q in queries:
            vec = embedder.embed_query(q)
            df = tbl.search(vec).limit(fetch_limit).to_pandas()
            if source_filter:
                df = df[df["source"].isin(source_filter)]
            for _, row in df.iterrows():
                dist = float(row.get("_distance", 1.0))
                rid = row["id"]
                if rid not in best or dist < best[rid][0]:
                    best[rid] = (dist, row)
        return [row for _, row in sorted(best.values(), key=lambda x: x[0])]

    def _fts(self, tbl, queries, source_filter, fetch_limit) -> list:
        """Multi-query FTS; merge with RRF (rank-based fusion)."""
        K_RRF = 60
        rrf: dict[str, float] = defaultdict(float)
        rows_by_id: dict[str, object] = {}
        for q in queries:
            try:
                df = tbl.search(q, query_type="fts").limit(fetch_limit).to_pandas()
            except Exception:
                continue
            if source_filter:
                df = df[df["source"].isin(source_filter)]
            for rank, (_, row) in enumerate(df.iterrows(), 1):
                rid = row["id"]
                rrf[rid] += 1.0 / (K_RRF + rank)
                if rid not in rows_by_id:
                    rows_by_id[rid] = row
        return [rows_by_id[rid] for rid in sorted(rrf, key=rrf.__getitem__, reverse=True)]

    def _hybrid(self, tbl, queries, source_filter, fetch_limit) -> list:
        """Multi-query hybrid (vector + FTS with RRF); merge by best relevance score."""
        from lancedb.rerankers import RRFReranker

        rrf_reranker = RRFReranker()
        best: dict[str, tuple[float, object]] = {}
        for q in queries:
            vec = embedder.embed_query(q)
            try:
                df = (
                    tbl.search(query_type="hybrid")
                    .text(q)
                    .vector(vec)
                    .rerank(rrf_reranker)
                    .limit(fetch_limit)
                    .to_pandas()
                )
            except Exception:
                # FTS index missing or query unsupported — fall back to vector
                df = tbl.search(vec).limit(fetch_limit).to_pandas()
                if source_filter:
                    df = df[df["source"].isin(source_filter)]
                for _, row in df.iterrows():
                    dist = float(row.get("_distance", 1.0))
                    rid = row["id"]
                    if rid not in best or dist < best[rid][0]:
                        best[rid] = (dist, row)
                continue

            if source_filter:
                df = df[df["source"].isin(source_filter)]
            for _, row in df.iterrows():
                score = float(row.get("_relevance_score", 0.0))
                rid = row["id"]
                if rid not in best or score > best[rid][0]:
                    best[rid] = (score, row)

        return [row for _, row in sorted(best.values(), key=lambda x: x[0], reverse=True)]


retriever = KnowledgeRetriever()
