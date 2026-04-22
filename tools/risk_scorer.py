import json
from typing import Union
from langchain.tools import tool

SCORE_THRESHOLDS = {
    "low":    (0, 3),
    "medium": (4, 6),
    "high":   (7, 10),
}


def compute_risk_score(patient: dict) -> dict:
    score = 0
    factors = []
    breakdown = {}

    try:
        age = int(patient.get("age", 0))
    except (ValueError, TypeError):
        age = 0

    if age >= 75:
        score += 2
        factors.append("Age >= 75")
        breakdown["age"] = 2
    else:
        breakdown["age"] = 0

    try:
        prior = int(patient.get("prior_admissions_6mo", 0))
    except (ValueError, TypeError):
        prior = 0

    if prior >= 2:
        score += 2
        factors.append("Prior admissions >= 2 in last 6 months")
        breakdown["prior_admissions"] = 2
    else:
        breakdown["prior_admissions"] = 0

    try:
        los = int(patient.get("length_of_stay_days", 0))
    except (ValueError, TypeError):
        los = 0

    if los > 7:
        score += 1
        factors.append("Length of stay > 7 days")
        breakdown["length_of_stay"] = 1
    else:
        breakdown["length_of_stay"] = 0

    follow_up = str(patient.get("follow_up_scheduled", "No")).strip().lower()
    if follow_up != "yes":
        score += 2
        factors.append("No follow-up scheduled")
        breakdown["no_follow_up"] = 2
    else:
        breakdown["no_follow_up"] = 0

    try:
        comorbidities = int(patient.get("comorbidity_count", 0))
    except (ValueError, TypeError):
        comorbidities = 0

    if comorbidities >= 3:
        score += 1
        factors.append("Comorbidity count >= 3")
        breakdown["comorbidities"] = 1
    else:
        breakdown["comorbidities"] = 0

    try:
        meds = int(patient.get("medication_count", 0))
    except (ValueError, TypeError):
        meds = 0

    if meds >= 8:
        score += 1
        factors.append("Medication count >= 8")
        breakdown["medication_count"] = 1
    else:
        breakdown["medication_count"] = 0

    discharge = str(patient.get("discharge_destination", "")).strip().lower()
    if discharge == "home":
        score += 1
        factors.append("Discharged to Home (no support services)")
        breakdown["discharge_home"] = 1
    else:
        breakdown["discharge_home"] = 0

    if score <= 3:
        risk_level = "Low"
    elif score <= 6:
        risk_level = "Medium"
    else:
        risk_level = "High"

    return {
        "risk_level": risk_level,
        "score": score,
        "contributing_factors": factors,
        "score_breakdown": breakdown,
    }


@tool
def risk_scorer_tool(patient_json: str) -> str:
    """Computes the readmission risk score and level from patient data via JSON string."""
    try:
        patient = json.loads(patient_json)
    except json.JSONDecodeError as exc:
        return json.dumps({"error": f"Invalid JSON input: {exc}"})

    result = compute_risk_score(patient)
    return json.dumps(result, indent=2)