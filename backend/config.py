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


settings = Settings()
