"""Deterministic clinical calculators used by screening and treatment specialists."""
from __future__ import annotations

# ── Cognitive assessment ───────────────────────────────────────────────────────

def interpret_cdr(score: float) -> str:
    labels = {
        0.0: "No impairment",
        0.5: "Questionable / very mild dementia",
        1.0: "Mild dementia",
        2.0: "Moderate dementia",
        3.0: "Severe dementia",
    }
    return labels.get(score, f"Unrecognised CDR score: {score}")


def interpret_mmse(score: int) -> str:
    if score >= 24:
        return "Normal cognition (≥24)"
    if score >= 18:
        return "Mild cognitive impairment (18–23)"
    if score >= 10:
        return "Moderate dementia (10–17)"
    return "Severe dementia (<10)"


def interpret_moca(score: int) -> str:
    if score >= 26:
        return "Normal cognition (≥26)"
    if score >= 18:
        return "Mild cognitive impairment (18–25)"
    return "Moderate-to-severe impairment (<18)"


def calculate_mmse(responses: dict[str, int]) -> int:
    """Sum raw item scores; caller supplies validated responses keyed by domain."""
    raise NotImplementedError


def calculate_moca(responses: dict[str, int]) -> int:
    raise NotImplementedError


def calculate_adas_cog(responses: dict[str, float]) -> float:
    raise NotImplementedError


# ── Dosing helpers ─────────────────────────────────────────────────────────────

def donepezil_dose(
    phase: str,              # "initiation" | "maintenance"
    severe: bool = False,
) -> dict[str, str]:
    """
    Returns recommended dose range per AWMF 038-013 / FDA labelling.
    severe=True applies the 23 mg option approved for severe AD.
    """
    if phase == "initiation":
        return {"dose": "5 mg/day", "duration": "4–6 weeks before uptitration"}
    if severe:
        return {"dose": "10–23 mg/day", "note": "23 mg only after ≥3 months on 10 mg"}
    return {"dose": "10 mg/day"}


def memantine_dose(phase: str) -> dict[str, str]:
    if phase == "initiation":
        return {"dose": "5 mg/day", "titration": "+5 mg/week over 4 weeks"}
    return {"dose": "20 mg/day (10 mg bid or 28 mg XR)"}
