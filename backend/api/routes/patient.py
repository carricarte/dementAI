import csv
from pathlib import Path

from fastapi import APIRouter, HTTPException

from backend.api.models import MRISummary, PatientResponse, ResearchSummary
from backend.state.store import patient_store

_SYNTHETIC_DIR = Path("data/synthetic")

router = APIRouter()


def _float(val: str) -> float | None:
    try:
        return float(val) if val else None
    except ValueError:
        return None


def _int(val: str) -> int | None:
    try:
        return int(val) if val else None
    except ValueError:
        return None


def _csv_row(sheet: str, patient_id: str) -> dict | None:
    path = _SYNTHETIC_DIR / f"{sheet}.csv"
    if not path.exists():
        return None
    with open(path, newline="", encoding="utf-8") as f:
        for row in csv.DictReader(f):
            if row.get("patient_uuid") == patient_id:
                return dict(row)
    return None


@router.get("/{patient_id}/research", response_model=ResearchSummary)
def get_patient_research(patient_id: str) -> ResearchSummary:
    uds = _csv_row("nacc_uds", patient_id)
    if uds is None:
        raise HTTPException(status_code=404, detail="No research data for this patient")

    mri_row = _csv_row("scan_mri", patient_id)
    gen_row = _csv_row("genetics", patient_id)

    mri = None
    if mri_row:
        mri = MRISummary(
            field_T=_float(mri_row.get("mri_field_T", "")),
            hippo_l_mm3=_float(mri_row.get("hippl_mm3", "")),
            hippo_r_mm3=_float(mri_row.get("hippr_mm3", "")),
            wmh_cm3=_float(mri_row.get("wmh_cm3", "")),
            mta_l=_float(mri_row.get("mta_score_l", "")),
            mta_r=_float(mri_row.get("mta_score_r", "")),
            amyloid_status=mri_row.get("amyloid_status") or None,
        )

    sex_code = uds.get("sex", "")
    sex = "F" if sex_code == "2" else ("M" if sex_code == "1" else None)

    return ResearchSummary(
        naccid=uds.get("naccid", patient_id),
        visit_date=uds.get("visit_date") or None,
        phenotype=uds.get("phenotype") or None,
        sex=sex,
        age=_int(uds.get("age", "")),
        cdr=_float(uds.get("cdrglob", "")),
        cdrsb=_float(uds.get("cdrsb", "")),
        mmse=_int(uds.get("mmse", "")),
        moca=_int(uds.get("moca", "")),
        gds=_int(uds.get("gds", "")),
        apoe_genotype=gen_row.get("apoe_genotype") if gen_row else None,
        apoe_e4_count=_int(gen_row.get("apoe_e4_count", "")) if gen_row else None,
        mri=mri,
    )


@router.get("/{patient_id}", response_model=PatientResponse)
def get_patient(patient_id: str) -> PatientResponse:
    record = patient_store.load(patient_id)
    if record is None:
        raise HTTPException(status_code=404, detail="Patient not found")
    return PatientResponse(record=record)


@router.delete("/{patient_id}")
def delete_patient(patient_id: str) -> dict:
    p = patient_store._path(patient_id)
    if not p.exists():
        raise HTTPException(status_code=404, detail="Patient not found")
    p.unlink()
    return {"deleted": patient_id}
