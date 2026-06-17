# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Commands

```bash
# Backend dev server (port 8000, auto-reload)
.venv/bin/uvicorn backend.main:app --reload --port 8000

# Frontend dev server (port 3000)
cd frontend && npm run dev

# Populate knowledge base (downloads AWMF PDF, embeds ~2 771 chunks — ~5 min first run)
python scripts/ingest.py --source awmf

# Tests
pytest                          # all tests
pytest tests/test_coordinator.py -k test_classify_stage_screening

# Lint / format
ruff check .
ruff format .

# Type check
mypy backend
```

Install extras when needed:
```bash
pip install -e ".[dev]"      # pytest, ruff, mypy
pip install -e ".[rag]"      # sentence-transformers, torch (needed for embedder)
pip install -e ".[ingest]"   # pypdf, beautifulsoup4 (needed for ingest scripts)
```

## Architecture

### Request flow

```
POST /query  →  coordinator (LangGraph)
                  │
                  ├─ classify_stage  (LLM classifies query → ClinicalStage enum)
                  ├─ load_patient    (JSON from data/patients/)
                  ├─ route_to_specialist  (conditional edge on stage value)
                  │       ↓ one of: screening / diagnosis / prevention / treatment / care
                  │   retrieve(query, source_filter)  →  LanceDB vector search
                  │   get_llm().invoke(prompt)         →  Claude Opus 4.8
                  ├─ merge_output    (passthrough; extend here for multi-specialist)
                  ├─ save_state      (appends VisitRecord to PatientRecord JSON)
                  └─ audit_log       (append-only JSONL in data/audit/)
```

### Key modules

| Path | Role |
|---|---|
| `backend/state/schema.py` | All shared types: `GraphState` (LangGraph dict), `PatientRecord`, `VisitRecord`, `Citation`, `ClinicalStage`, `DementiaType` |
| `backend/agents/coordinator.py` | Builds and compiles the LangGraph `StateGraph`; routing logic lives here |
| `backend/agents/{screening,diagnosis,prevention,treatment,care}.py` | Each specialist: calls `retrieve()` with its own `source_filter`, builds a single prompt, calls `get_llm()` |
| `backend/rag/retriever.py` | `KnowledgeRetriever`: embeds query → LanceDB ANN search → deduped `Citation` list |
| `backend/rag/ingestion.py` | `Document` dataclass + `ingest()` — common entry point for all sources |
| `backend/rag/embedder.py` | Wraps `sentence-transformers` (`neuml/pubmedbert-base-embeddings`, 768-dim) |
| `backend/config.py` | `Settings` (pydantic-settings); reads `.env`; controls model, paths, `retrieval_top_k` |
| `backend/llm.py` | `get_llm()` — `lru_cache(1)` singleton `ChatAnthropic` |
| `backend/state/store.py` | `PatientStore` — JSON persistence under `data/patients/` |
| `backend/audit/logger.py` | Append-only JSONL audit writer |
| `backend/api/` | FastAPI routers (`/query`, `/patient`) + Pydantic request/response models |
| `frontend/src/App.tsx` | All React state; owns patient ID, record, response, selectedVisit |
| `frontend/src/api/client.ts` | `fetchPatient()`, `submitQuery()` — thin fetch wrappers |

### Data persistence

All runtime data lands under `data/` (gitignored except `.gitkeep`):
- `data/lancedb/` — LanceDB vector store (table: `knowledge`)
- `data/patients/` — one JSON file per patient ID
- `data/audit/` — append-only JSONL audit log

### Adding a knowledge source

1. Create `backend/rag/sources/{source}.py` with a `fetch() -> list[Document]` function
2. Wire it into `scripts/ingest.py` with a `--source {source}` branch
3. Run `python scripts/ingest.py --source {source}`

Valid `source` values for `source_filter` in specialists: `awmf`, `pubmed`, `clinicaltrials`, `neurology`, `alz`, `aan`.