from pydantic_settings import BaseSettings, SettingsConfigDict


class Settings(BaseSettings):
    model_config = SettingsConfigDict(env_file=".env", extra="ignore")

    anthropic_api_key: str = ""
    claude_model: str = "claude-opus-4-8"

    lancedb_path: str = "data/lancedb"
    patient_store_path: str = "data/patients"
    audit_log_path: str = "data/audit"

    # Embedding model used for RAG (sentence-transformers model ID)
    # Default: neuml/pubmedbert-base-embeddings — public, 768-dim, biomedical retrieval
    # Alternative: ncats/MedCPT-Query-Encoder (requires HF org authentication)
    embed_model: str = "neuml/pubmedbert-base-embeddings"
    retrieval_top_k: int = 20

    # Max chunks allowed from the same paper after reranking.
    # Prevents the reranker from filling all top-k slots with chunks from one
    # highly-relevant paper, which would produce very few unique references.
    max_chunks_per_paper: int = 2

    # Search strategy: vector | fts | hybrid | hybrid_rerank
    # hybrid_rerank = hybrid (vector+FTS RRF) then cross-encoder rerank
    search_strategy: str = "hybrid_rerank"
    # Cross-encoder model for reranking (sentence-transformers model ID).
    # ncats/MedCPT-Cross-Encoder is trained on MEDLINE query-article pairs —
    # significantly better than MS MARCO models for biomedical literature.
    rerank_model: str = "ncats/MedCPT-Cross-Encoder"


settings = Settings()
