from fastapi import APIRouter, HTTPException
from fastapi.responses import StreamingResponse

from backend.agents.coordinator import coordinator, stream_query
from backend.api.models import QueryRequest, QueryResponse
from backend.state.schema import GraphState

router = APIRouter()


@router.post("/stream")
def handle_query_stream(req: QueryRequest):
    return StreamingResponse(
        stream_query(req.patient_id, req.query),
        media_type="text/event-stream",
        headers={"Cache-Control": "no-cache", "X-Accel-Buffering": "no"},
    )


@router.post("/", response_model=QueryResponse)
def handle_query(req: QueryRequest) -> QueryResponse:
    initial: GraphState = {
        "patient_id": req.patient_id,
        "query": req.query,
        "is_on_topic": True,
        "stage": None,
        "query_intent": None,
        "patient_record": None,
        "patient_status_report": None,
        "specialist_response": None,
        "citations": [],
        "final_response": None,
    }
    try:
        result = coordinator.invoke(initial)
    except Exception as exc:
        raise HTTPException(status_code=500, detail=str(exc)) from exc
    return QueryResponse(
        patient_id=req.patient_id,
        stage=result["stage"],
        response=result["final_response"] or "",
        citations=result["citations"],
        personalized=result.get("query_intent") == "patient_specific",
    )
