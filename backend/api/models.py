from pydantic import BaseModel

from backend.state.schema import Citation, ClinicalStage, PatientRecord


class QueryRequest(BaseModel):
    patient_id: str | None = None
    query: str


class QueryResponse(BaseModel):
    patient_id: str | None
    stage: ClinicalStage
    response: str
    citations: list[Citation]
    personalized: bool = False


class PatientResponse(BaseModel):
    record: PatientRecord


class MRISummary(BaseModel):
    field_T: float | None = None
    hippo_l_mm3: float | None = None
    hippo_r_mm3: float | None = None
    wmh_cm3: float | None = None
    mta_l: float | None = None
    mta_r: float | None = None
    amyloid_status: str | None = None


class ResearchSummary(BaseModel):
    naccid: str
    visit_date: str | None = None
    phenotype: str | None = None
    sex: str | None = None  # "M" | "F"
    age: int | None = None
    cdr: float | None = None
    cdrsb: float | None = None
    mmse: int | None = None
    moca: int | None = None
    gds: int | None = None
    apoe_genotype: str | None = None
    apoe_e4_count: int | None = None
    mri: MRISummary | None = None
