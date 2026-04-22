from __future__ import annotations

import sys
if sys.version_info >= (3, 14):
    try:
        import pydantic._internal._typing_extra as _typing_extra

        _orig_try_eval = _typing_extra.try_eval_type
        _orig_eval_bp  = _typing_extra.eval_type_backport

        def _patched_try_eval(value, globalns, localns):
            try:
                return _orig_try_eval(value, globalns, localns)
            except TypeError as exc:
                if "'function' object is not subscriptable" in str(exc):
                    return value, False
                raise

        def _patched_eval_bp(value, globalns=None, localns=None, type_params=None):
            try:
                return _orig_eval_bp(value, globalns, localns, type_params)
            except TypeError as exc:
                if "'function' object is not subscriptable" in str(exc):
                    return value
                raise

        _typing_extra.try_eval_type      = _patched_try_eval
        _typing_extra.eval_type_backport = _patched_eval_bp
    except Exception:
        pass

import json
import os
import re
from typing import Any, Optional

from dotenv import load_dotenv
load_dotenv()

from langchain.tools import tool
from langchain_ollama import ChatOllama

MODEL_NAME      = os.getenv("OLLAMA_MODEL")
OLLAMA_BASE_URL = os.getenv("OLLAMA_BASE_URL")

if not MODEL_NAME:
    raise EnvironmentError("OLLAMA_MODEL is not set. Add it to your .env file.")
if not OLLAMA_BASE_URL:
    raise EnvironmentError("OLLAMA_BASE_URL is not set. Add it to your .env file.")

from tools.risk_scorer import compute_risk_score, risk_scorer_tool
from utils.data_loader import load_patient_by_id

_PROMPT_PATH = os.path.join(
    os.path.dirname(os.path.abspath(__file__)), "prompts", "system_prompt.txt"
)

_CONDITION_MAP = {
    "heart failure": "Heart Failure",
    "hf": "Heart Failure",
    "chf": "Heart Failure",
    "copd": "COPD",
    "chronic obstructive": "COPD",
    "diabetes": "Diabetes",
    "diabetic": "Diabetes",
    "dm": "Diabetes",
    "pneumonia": "Pneumonia",
    "hip fracture": "Hip Fracture",
    "hip": "Hip Fracture",
}

_ROLE_LABELS = {
    "care coordinators": "Care Coordinators",
    "hospital operations team": "Hospital Operations Team",
    "case managers": "Case Managers",
    "nursing staff": "Nursing Staff",
}

_DEFAULT_ROLE = "Care Coordinators"

_SAFETY_DISCLAIMER = (
    "SAFETY DISCLAIMER: This is a decision-support tool. "
    "Always consult a licensed clinician before acting on this output."
)

_FALLBACK_WARNING_PREFIX = (
    "LLM reasoning unavailable. Deterministic explanation mode used:"
)

_MEDICAL_KEYWORDS = {
    "patient", "readmission", "risk", "discharge", "hospital", "condition",
    "diagnosis", "age", "stay", "admission", "follow", "medication", "med",
    "comorbid", "rehab", "snf", "nursing", "heart", "copd", "diabetes",
    "pneumonia", "fracture", "assess", "score", "high", "medium", "low",
    "care", "coordinator", "case", "manager", "clinical", "health",
}


def _is_medical_query(text: str) -> bool:
    """Return True only if the query looks medical/patient-related."""
    if re.search(r"\bP\d{3}\b", text.upper()):
        return True
    words = set(re.findall(r"[a-z]+", text.lower()))
    return bool(words & _MEDICAL_KEYWORDS)


def _load_system_prompt() -> str:
    if not os.path.exists(_PROMPT_PATH):
        raise FileNotFoundError(f"System prompt not found at: {_PROMPT_PATH}")
    with open(_PROMPT_PATH, "r", encoding="utf-8") as f:
        return f.read()

@tool
def patient_lookup_tool(patient_id: str) -> str:
    """Looks up patient details by patient ID."""
    patient_id = str(patient_id).strip().upper()
    record = load_patient_by_id(patient_id)
    if record is None:
        return json.dumps({
            "error": f"Patient '{patient_id}' not found in the dataset.",
            "hint": "Valid IDs are P001 through P040."
        })
    return json.dumps(record, default=str)


def parse_conversational_input(text: str) -> dict:
    t = text.lower()

    age = 0
    m = re.search(r'(\d{1,3})\s*(?:year[s]?\s*old|yo\b|-year)', t)
    if m:
        age = int(m.group(1))

    condition = "Unknown"
    for keyword, canonical in _CONDITION_MAP.items():
        if keyword in t:
            condition = canonical
            break

    los = 0
    m = re.search(r'(\d+)\s*(?:-?\s*day[s]?\s*stay|days?\s+(?:in\s+)?hospital|day\s+stay)', t)
    if m:
        los = int(m.group(1))

    prior = 0
    m = re.search(
        r'(\d+)\s*(?:prior|previous|past|recent)?\s*admissions?'
        r'(?:\s*(?:in|over|within)\s*(?:the\s*)?(?:last\s*)?(?:6|six)\s*months?)?',
        t
    )
    if m:
        prior = int(m.group(1))

    follow_up = "Yes"
    no_followup_patterns = [
        r'no\s+follow.?up',
        r'follow.?up\s+(?:not|hasn\'t been|hasn\'t|not\s+been)\s*(?:booked|scheduled|arranged|set)',
        r'(?:not|no)\s+(?:booked|scheduled|arranged)\s+(?:follow.?up|appointment)',
        r'without\s+follow.?up',
        r'follow.?up\s+(?:missing|absent|lacking)',
    ]
    for pat in no_followup_patterns:
        if re.search(pat, t):
            follow_up = "No"
            break

    meds = 0
    m = re.search(r'(\d+)\s*(?:medications?|meds?|drugs?|prescriptions?)', t)
    if m:
        meds = int(m.group(1))

    comorbidities = 0
    m = re.search(r'(\d+)\s*(?:comorbidities|comorbidity|co-morbidities|conditions?)', t)
    if m:
        comorbidities = int(m.group(1))

    discharge = "Home"
    if re.search(r'\b(?:snf|nursing\s+facility|skilled\s+nursing)\b', t):
        discharge = "SNF"
    elif re.search(r'\brehab(?:ilitation)?\b', t):
        discharge = "Rehab"
    elif re.search(r'\bhome\s*(?:\+|with|and)\s*services?\b', t):
        discharge = "Home+Services"
    elif re.search(r'\bhome\b', t):
        discharge = "Home"

    return {
        "patient_id": "CONV",
        "age": age,
        "primary_condition": condition,
        "length_of_stay_days": los,
        "prior_admissions_6mo": prior,
        "discharge_destination": discharge,
        "follow_up_scheduled": follow_up,
        "medication_count": meds,
        "comorbidity_count": comorbidities,
    }


def _normalize_role(role: Optional[str]) -> str:
    if not role:
        return _DEFAULT_ROLE
    normalized = str(role).strip().lower()
    return _ROLE_LABELS.get(normalized, _DEFAULT_ROLE)


def _call_risk_scorer_tool(patient: dict) -> dict:
    patient_json = json.dumps(patient, default=str)
    raw = risk_scorer_tool.func(patient_json)
    result = json.loads(raw)
    if "error" in result:
        raise ValueError(result["error"])
    return result


def _extract_first_json_object(text: str) -> dict:
    cleaned = re.sub(r"```(?:json)?", "", text, flags=re.IGNORECASE).strip()
    match = re.search(r"\{.*\}", cleaned, re.DOTALL)
    if not match:
        raise ValueError("No JSON object found in LLM response")
    return json.loads(match.group())


def _fallback_actions(risk: dict, role: str) -> list[str]:
    factors = risk.get("contributing_factors", [])
    level = risk.get("risk_level", "Low")
    actions = []

    if "No follow-up scheduled" in factors:
        actions.append("Schedule follow-up within 7 days and confirm patient receives appointment details.")
    if "Prior admissions >= 2 in last 6 months" in factors:
        actions.append("Arrange proactive outreach calls in the first 72 hours after discharge.")
    if "Length of stay > 7 days" in factors:
        actions.append("Create a discharge transition checklist and verify completion before handoff.")
    if "Medication count >= 8" in factors:
        actions.append("Assign medication reconciliation review and confirm understanding at discharge.")

    if role == "Hospital Operations Team":
        actions.append("Flag this case in daily operations huddle for transition-of-care capacity planning.")
    elif role == "Case Managers":
        actions.append("Coordinate post-discharge support services and document barriers to follow-up.")
    elif role == "Nursing Staff":
        actions.append("Use a discharge education checklist and confirm teach-back completion.")
    else:
        actions.append("Confirm care coordinator ownership and close-loop follow-up tracking.")

    if level == "High":
        actions.insert(0, "ESCALATE: Immediate care coordinator review recommended.")

    deduped = []
    seen = set()
    for item in actions:
        if item not in seen:
            deduped.append(item)
            seen.add(item)
    return deduped[:4]


def _render_assessment_response(
    patient_id: str,
    role: str,
    risk: dict,
    reasoning_summary: str,
    actions: list[str],
    llm_warning: Optional[str] = None,
) -> str:
    badge = {
        "High": "HIGH",
        "Medium": "MEDIUM",
        "Low": "LOW",
    }.get(risk.get("risk_level", "Low"), "LOW")

    factors = risk.get("contributing_factors") or ["No major risk factors identified by scoring rules."]
    top_factors = factors[:3]

    if risk.get("risk_level") == "High" and not any("ESCALATE" in a for a in actions):
        actions = ["ESCALATE: Immediate care coordinator review recommended."] + actions

    lines = [
        "READMISSION RISK ASSESSMENT",
        f"Target User: {role}",
        f"Patient ID: {patient_id}",
        f"Readmission Risk Level: {badge}",
        f"Risk Score: {risk.get('score', 0)} / 10",
        "",
        "REASONING SUMMARY:",
        reasoning_summary.strip() or "Risk interpretation generated from patient profile and scoring output.",
        "",
        "TOP CONTRIBUTING FACTORS:",
    ]

    for factor in top_factors:
        lines.append(f"- {factor}")

    lines.extend(["", "RECOMMENDED PREVENTIVE ACTIONS:"])
    for action in actions[:4]:
        lines.append(f"- {action}")

    if llm_warning:
        lines.extend(["", llm_warning])

    lines.extend(["", _SAFETY_DISCLAIMER])
    return "\n".join(lines)


def _build_reasoning_prompt(
    patient: dict,
    risk: dict,
    role: str,
    query: str,
) -> str:
    return (
        "You are a non-clinical hospital discharge decision-support assistant.\n"
        "Return JSON only with this exact schema:\n"
        "{\n"
        "  \"reasoning_summary\": string,\n"
        "  \"preventive_actions\": [string, string, string],\n"
        "  \"top_contributing_factors\": [string, string, string]\n"
        "}\n"
        "Rules:\n"
        "- Use non-clinical, coordination-focused actions only.\n"
        "- Do not diagnose or prescribe medications.\n"
        "- If risk_level is High, include this exact action: \"ESCALATE: Immediate care coordinator review recommended.\"\n"
        "- Tailor wording for this target user role: "
        f"{role}.\n\n"
        f"Original request: {query or 'N/A'}\n\n"
        f"Patient data:\n{json.dumps(patient, indent=2)}\n\n"
        f"Risk assessment:\n{json.dumps(risk, indent=2)}"
    )


def _llm_reasoning_payload(patient: dict, risk: dict, role: str, query: str) -> dict:
    llm = ChatOllama(
        model=MODEL_NAME,
        base_url=OLLAMA_BASE_URL,
        temperature=0.1,
        num_predict=900,
    )
    response = llm.invoke(_build_reasoning_prompt(patient, risk, role, query))
    content = getattr(response, "content", response)

    if isinstance(content, list):
        pieces = []
        for chunk in content:
            if isinstance(chunk, dict):
                pieces.append(str(chunk.get("text", "")))
            else:
                pieces.append(str(chunk))
        text = "\n".join(pieces).strip()
    else:
        text = str(content).strip()

    payload = _extract_first_json_object(text)
    if not isinstance(payload, dict):
        raise ValueError("Unexpected LLM reasoning payload type")
    return payload


def run_assessment_with_reasoning(
    query: str = "",
    role: str = _DEFAULT_ROLE,
    patient_id: Optional[str] = None,
    allow_llm: bool = True,
) -> dict[str, Any]:
    role_name = _normalize_role(role)

    patient: Optional[dict] = None
    source_query = (query or "").strip()

    if patient_id:
        pid = str(patient_id).strip().upper()
        patient = load_patient_by_id(pid)
        if patient is None:
            return {
                "error": f"Patient '{pid}' not found.",
                "response_text": (
                    f"READMISSION RISK ASSESSMENT\n"
                    f"Target User: {role_name}\n"
                    f"Patient ID: {pid}\n\n"
                    f"Unable to locate this patient in the local dataset.\n\n"
                    f"{_SAFETY_DISCLAIMER}"
                ),
                "patient_data": None,
                "risk_assessment": None,
                "used_llm_reasoning": False,
                "used_fallback": True,
                "role": role_name,
            }
    else:
        pid_match = re.search(r"\bP\d+\b", source_query.upper())
        if pid_match:
            patient = load_patient_by_id(pid_match.group())
            if patient is None:
                # ID not in CSV — parse the rest of the message
                # conversationally and use the mentioned ID as the label
                patient = parse_conversational_input(source_query)
                patient["patient_id"] = pid_match.group()
        elif source_query:
            if not _is_medical_query(source_query):
                return {
                    "error": None,
                    "response_text": (
                        "I'm a clinical decision-support assistant specialised in "
                        "30-day hospital readmission risk.\n\n"
                        "I can only help with medical or patient-related queries. "
                        "Please enter a patient ID (e.g. P007) or describe a "
                        "patient profile to get a readmission risk assessment."
                    ),
                    "patient_data": None,
                    "risk_assessment": None,
                    "used_llm_reasoning": False,
                    "used_fallback": False,
                    "role": role_name,
                }
            patient = parse_conversational_input(source_query)

    if patient is None:
        return {
            "error": "No patient information was provided.",
            "response_text": (
                "READMISSION RISK ASSESSMENT\n\n"
                "Provide a patient ID (for example, P007) or describe a patient profile "
                "to run an assessment.\n\n"
                f"{_SAFETY_DISCLAIMER}"
            ),
            "patient_data": None,
            "risk_assessment": None,
            "used_llm_reasoning": False,
            "used_fallback": True,
            "role": role_name,
        }

    patient.setdefault("patient_id", patient_id or "NEW")
    patient_id_text = str(patient.get("patient_id", "NEW")).strip() or "NEW"

    try:
        risk = _call_risk_scorer_tool(patient)
    except Exception as exc:
        return {
            "error": str(exc),
            "response_text": (
                f"READMISSION RISK ASSESSMENT\n"
                f"Target User: {role_name}\n"
                f"Patient ID: {patient_id_text}\n\n"
                f"Risk scoring tool failed: {exc}\n\n"
                f"{_SAFETY_DISCLAIMER}"
            ),
            "patient_data": patient,
            "risk_assessment": None,
            "used_llm_reasoning": False,
            "used_fallback": True,
            "role": role_name,
        }

    reasoning_summary = (
        "Patient profile and rule-based score were reviewed to estimate readmission risk and transition needs."
    )
    actions = _fallback_actions(risk, role_name)
    llm_warning = None
    used_llm = False
    used_fallback = True

    if allow_llm:
        try:
            payload = _llm_reasoning_payload(patient, risk, role_name, source_query)

            candidate_summary = str(payload.get("reasoning_summary", "")).strip()
            if candidate_summary:
                reasoning_summary = candidate_summary

            candidate_actions = payload.get("preventive_actions", [])
            if isinstance(candidate_actions, list):
                cleaned_actions = [str(a).strip() for a in candidate_actions if str(a).strip()]
                if cleaned_actions:
                    actions = cleaned_actions[:4]

            candidate_factors = payload.get("top_contributing_factors", [])
            if isinstance(candidate_factors, list):
                cleaned_factors = [str(f).strip() for f in candidate_factors if str(f).strip()]
                if cleaned_factors:
                    risk["contributing_factors"] = cleaned_factors

            used_llm = True
            used_fallback = False
        except Exception as exc:
            llm_warning = f"{_FALLBACK_WARNING_PREFIX} {exc}"

    response_text = _render_assessment_response(
        patient_id=patient_id_text,
        role=role_name,
        risk=risk,
        reasoning_summary=reasoning_summary,
        actions=actions,
        llm_warning=llm_warning,
    )

    return {
        "error": None,
        "response_text": response_text,
        "patient_data": patient,
        "risk_assessment": risk,
        "used_llm_reasoning": used_llm,
        "used_fallback": used_fallback,
        "role": role_name,
    }