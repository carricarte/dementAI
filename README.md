# DementIA — AI-powered dementia clinical decision support

A multi-agent clinical decision support system for dementia care. A coordinator classifies each query by clinical stage and intent, then routes it down one of two paths: a general evidence path (knowledge base + specialist agent) or a patient-specific path (Analyzer Agent → synthesis).

## Architecture

```
POST /query  or  POST /query/stream
        │
        ▼
  classify_stage          LLM → ClinicalStage (screening / diagnosis /
        │                        prevention / treatment / care)
        ▼
  classify_intent         LLM → "general" or "patient_specific"
        │                 (always "general" when no patient_id supplied)
        │
   ┌────┴────────────────────────────────────┐
   │ general                  patient_specific│
   ▼                                         ▼
load_patient               Analyzer Agent
   │                       reads PatientStore + NACC UDS CSVs
   │                       → PatientStatusReport (structured JSON)
   ▼                                         │
specialist (by stage)                        ▼
  ┌──────────┬─────────────┬──────┬──────┐  synthesize_patient_specific
  │Screening │  Diagnosis  │Prev. │Treat.│  merges PatientStatusReport
  │          │             │      │      │  + KB evidence → personalized
  └──────────┴──────┬──────┴──────┴──────┘  response with citations
                    │                                │
                    └──────────────┬─────────────────┘
                                   ▼
                            merge_output
                                   │
                             save_state       append VisitRecord to
                                   │          PatientRecord JSON
                              audit_log       append-only JSONL
                                   │
                                  END
```

**Two paths, one SSE stream:**

| Event | Field |
|---|---|
| `{"type":"stage","stage":"..."}` | classified clinical stage |
| `{"type":"intent","intent":"..."}` | `"general"` or `"patient_specific"` |
| `{"type":"chunk","text":"..."}` | streaming response token |
| `{"type":"done","citations":[...],"personalized":bool}` | final metadata |

**Stack**

| Layer | Technology |
|---|---|
| LLM | Claude Opus 4.8 (Anthropic) |
| Agents | LangGraph StateGraph |
| Knowledge base | LanceDB + PubMedBERT embeddings (768-dim) |
| Research data | NACC UDS, MRI scan, genetics CSVs (synthetic) |
| API | FastAPI — `POST /query`, `POST /query/stream` |
| Frontend | React + Vite + Tailwind CSS |

## Agents

### Coordinator
Classifies the query by `ClinicalStage` and `query_intent`, then branches:
- **General path** — loads the patient record, delegates to the stage specialist, streams the KB-grounded response.
- **Patient-specific path** — invokes the Analyzer Agent first, then synthesizes a personalized response against KB evidence.

### Analyzer Agent (`backend/agents/analyzer.py`)
Invoked only on the `patient_specific` path. Reads structured patient data (clinical record from `PatientStore` + NACC UDS / MRI / genetics CSVs) and produces a `PatientStatusReport` with diagnosis stage, cognitive scores, risk factors, medications, MRI findings, and ranked treatment priorities. Patient data never enters the knowledge base.

### Specialist Agents
Each specialist retrieves from a stage-specific subset of the knowledge base and builds a grounded response.

| Stage | Specialist | Source filters |
|---|---|---|
| Screening | Screening | `aan`, `awmf`, `alz` |
| Diagnosis | Diagnosis | `pubmed`, `neurology`, `awmf` |
| Prevention | Prevention | `pubmed`, `alz` |
| Treatment | Treatment | `pubmed`, `clinicaltrials`, `awmf` |
| Care | Patient Care | `alz`, `aan` |

## Knowledge base

Currently populated:

| Source | Status | Chunks |
|---|---|---|
| AWMF S3-Leitlinie Demenzen (038-013, v6.1 2026) | ✅ | ~2,771 |
| PubMed (NCBI E-utilities) | ✅ | ~4,130 |
| ClinicalTrials.gov | not yet implemented | — |
| Neurology.org | not yet implemented | — |
| Alz.org | not yet implemented | — |
| AAN.com | not yet implemented | — |

## Setup

### Prerequisites

- Python ≥ 3.11
- Node.js ≥ 18
- An [Anthropic API key](https://console.anthropic.com/)

### 1. Clone and install

```bash
git clone git@github.com:carricarte/DementIA.git
cd DementIA

# Python backend
python -m venv .venv
source .venv/bin/activate
pip install -e ".[ingest,rag]"

# React frontend
cd frontend && npm install && cd ..
```

### 2. Configure environment

```bash
cp .env.example .env
# Edit .env and set ANTHROPIC_API_KEY=sk-ant-...
```

### 3. Populate the knowledge base

```bash
python scripts/ingest.py --source awmf
python scripts/ingest.py --source pubmed
```

Runtime: ~5 min on first run (downloads the PubMedBERT embedding model).

## Running

Start both servers in separate terminals:

```bash
# Backend (port 8000)
.venv/bin/uvicorn backend.main:app --reload --port 8000

# Frontend (port 3000)
cd frontend && npm run dev
```

Open **http://localhost:3000**, enter a patient ID, and submit a clinical query.

## Project structure

```
DementIA/
├── backend/
│   ├── agents/
│   │   ├── coordinator.py   # LangGraph graph; classify_stage → classify_intent → branch
│   │   ├── analyzer.py      # Analyzer Agent: PatientStore + NACC UDS → PatientStatusReport
│   │   ├── screening.py
│   │   ├── diagnosis.py
│   │   ├── prevention.py
│   │   ├── treatment.py
│   │   └── care.py
│   ├── api/
│   │   ├── routes/
│   │   │   ├── query.py     # POST /query, POST /query/stream
│   │   │   └── patient.py   # Patient CRUD
│   │   └── models.py        # QueryRequest, QueryResponse (personalized field)
│   ├── audit/               # Append-only JSONL audit writer
│   ├── rag/
│   │   ├── embedder.py      # PubMedBERT sentence-transformers wrapper
│   │   ├── retriever.py     # LanceDB vector search + enrich_query
│   │   ├── chunker.py       # Token-aware sentence chunker (512 tokens)
│   │   ├── ingestion.py     # Document dataclass + ingest()
│   │   └── sources/
│   │       ├── awmf.py      # AWMF REST API downloader + PDF chunker
│   │       └── pubmed.py    # NCBI E-utilities fetcher
│   ├── state/
│   │   ├── schema.py        # GraphState, PatientRecord, PatientStatusReport, Citation
│   │   └── store.py         # PatientStore — JSON persistence under data/patients/
│   ├── tools/
│   │   ├── calculators.py   # Screening score calculators (MMSE, MoCA, CDR…)
│   │   └── retrieval.py     # retrieve() + enrich_query() wrappers
│   ├── config.py            # Settings (pydantic-settings, reads .env)
│   ├── llm.py               # Cached ChatAnthropic singleton
│   └── main.py              # FastAPI app entry point
├── frontend/
│   └── src/
│       ├── App.tsx              # Root component; owns patient ID, record, response
│       ├── api/client.ts        # fetchPatient(), streamQuery() — SSE event handling
│       ├── types/index.ts       # TypeScript interfaces (mirrors Pydantic models)
│       └── components/
│           ├── QueryPanel.tsx   # Query input + streaming Markdown response
│           ├── VisitHistory.tsx # Left sidebar: past visits
│           ├── PatientProfile.tsx # Right sidebar: patient record
│           ├── CitationList.tsx # Numbered references section
│           └── StageBadge.tsx   # Clinical stage chip
├── scripts/
│   └── ingest.py            # CLI: populate LanceDB from each source
├── data/
│   ├── lancedb/             # Vector store (gitignored)
│   ├── patients/            # Patient JSON records (gitignored)
│   ├── audit/               # Audit JSONL (gitignored)
│   └── synthetic/           # NACC UDS / MRI / genetics CSVs (gitignored)
├── tests/
├── pyproject.toml
└── .env.example
```

## Adding a new knowledge source

1. Create `backend/rag/sources/{source}.py` with a `fetch() -> list[Document]` function
2. Wire it into `scripts/ingest.py`
3. Run `python scripts/ingest.py --source {source}`

Valid `source_filter` values: `awmf`, `pubmed`, `clinicaltrials`, `neurology`, `alz`, `aan`.

## Dementia types supported (20)

Alzheimer's · Vascular · Lewy body · FTD-behavioral · PPA semantic · PPA nonfluent · FTD-MND · Mixed · Parkinson's dementia · Huntington's · Corticobasal degeneration · PSP · Posterior cortical atrophy · LATE (TDP-43) · CTE · Creutzfeldt-Jakob · HIV-associated · Wernicke-Korsakoff · Normal pressure hydrocephalus · Down syndrome-associated
