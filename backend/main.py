from contextlib import asynccontextmanager

from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware

from backend.api.routes import patient, query


@asynccontextmanager
async def lifespan(app: FastAPI):
    from backend.rag.embedder import embedder

    embedder.warmup()
    yield


app = FastAPI(title="DementIA", version="0.1.0", lifespan=lifespan)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # tighten for production
    allow_methods=["*"],
    allow_headers=["*"],
)

app.include_router(query.router, prefix="/query", tags=["query"])
app.include_router(patient.router, prefix="/patient", tags=["patient"])


@app.get("/health")
def health() -> dict:
    return {"status": "ok"}
