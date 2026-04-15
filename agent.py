"""
agent.py — LangChain agent and orchestration for readmission risk assessment.

Interactive assessments use local LLM reasoning plus a rule-based scoring tool.
Dashboard and batch utilities can still use deterministic scoring directly.
"""

from __future__ import annotations

# Python 3.14 compatibility: patch pydantic's type evaluation
import sys
if sys.version_info >= (3, 14):
    try:
        import pydantic._internal._typing_extra as typing_extra

        _orig_try_eval_type = typing_extra.try_eval_type
        _orig_eval_type_backport = typing_extra.eval_type_backport

        def patched_try_eval_type(value, globalns, localns):
            """Wrapper that handles Python 3.14 annotation evaluation failures."""
            try:
                return _orig_try_eval_type(value, globalns, localns)
            except TypeError as exc:
                if "'function' object is not subscriptable" in str(exc):
                    return value, False
                raise

        def patched_eval_type_backport(value, globalns=None, localns=None, type_params=None):
            """Wrapper to catch remaining type evaluation crashes."""
            try:
                return _orig_eval_type_backport(value, globalns, localns, type_params)
            except TypeError as exc:
                if "'function' object is not subscriptable" in str(exc):
                    return value
                raise
        
        typing_extra.try_eval_type = patched_try_eval_type
        typing_extra.eval_type_backport = patched_eval_type_backport
    except Exception:
        pass

import json
import os
import re
from typing import Any, Optional, Union

from langchain.agents import AgentExecutor, create_react_agent
from langchain.prompts import PromptTemplate
from langchain.tools import tool
from langchain_ollama import ChatOllama

from tools.risk_scorer import compute_risk_score, risk_scorer_tool
from utils.data_loader import load_patient_by_id

# ── Path to system prompt ──────────────────────────────────────────────────────
_PROMPT_PATH = os.path.join(
    os.path.dirname(os.path.abspath(__file__)), "prompts", "system_prompt.txt"
)

# ── Condition synonym mapping ──────────────────────────────────────────────────
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
    "⚠️ SAFETY DISCLAIMER: This is a decision-support tool. "
    "Always consult a licensed clinician before acting on this output."
)

_FALLBACK_WARNING_PREFIX = (
    "⚠️ LLM reasoning unavailable. Deterministic explanation mode used:"
)


def _load_system_prompt() -> str:
    """Load the system prompt from the prompts directory.

    Returns:
        str: Raw system prompt text.

    Raises:
        FileNotFoundError: If system_prompt.txt is missing.
    """
    if not os.path.exists(_PROMPT_PATH):
        raise FileNotFoundError(f"System prompt not found at: {_PROMPT_PATH}")
    with open(_PROMPT_PATH, "r", encoding="utf-8") as f:
        return f.read()


# ── Tool: Patient Lookup ───────────────────────────────────────────────────────
@tool
def patient_lookup_tool(patient_id: str) -> str:
    """Look up a patient record in the hospital dataset by their patient ID.

    Args:
        patient_id (str): The patient identifier (e.g. 'P007').

    Returns:
        str: JSON-encoded patient record, or an error message if not found.
    """
    patient_id = str(patient_id).strip().upper()
    record = load_patient_by_id(patient_id)
    if record is None:
        return json.dumps({
            "error": f"Patient '{patient_id}' not found in the dataset.",
            "hint": "Valid IDs are P001 through P040."
        })
    return json.dumps(record, default=str)


# ── Free-text extraction (no LLM required) ────────────────────────────────────
def parse_conversational_input(text: str) -> dict:
    """Extract patient attributes from a free-text description.

    Supports phrases like:
      "65 year old with COPD, 3 prior admissions, no follow-up booked"
      "72yo female, heart failure, 13 day stay, 11 meds, 4 comorbidities"

    Applies conservative defaults for any field that cannot be parsed,
    so that assessments are always possible from partial information.

    Args:
        text (str): Raw natural-language patient description.

    Returns:
        dict: Partial patient record usable by compute_risk_score().
    """
    t = text.lower()

    # ── Age ───────────────────────────────────────────────────────────────────
    age = 0
    m = re.search(r'(\d{1,3})\s*(?:year[s]?\s*old|yo\b|-year)', t)
    if m:
        age = int(m.group(1))

    # ── Primary condition ─────────────────────────────────────────────────────
    condition = "Unknown"
    for keyword, canonical in _CONDITION_MAP.items():
        if keyword in t:
            condition = canonical
            break

    # ── Length of stay ────────────────────────────────────────────────────────
    los = 0
    m = re.search(r'(\d+)\s*(?:-?\s*day[s]?\s*stay|days?\s+(?:in\s+)?hospital|day\s+stay)', t)
    if m:
        los = int(m.group(1))

    # ── Prior admissions in last 6 months ─────────────────────────────────────
    prior = 0
    m = re.search(
        r'(\d+)\s*(?:prior|previous|past|recent)?\s*admissions?'
        r'(?:\s*(?:in|over|within)\s*(?:the\s*)?(?:last\s*)?(?:6|six)\s*months?)?',
        t
    )
    if m:
        prior = int(m.group(1))

    # ── Follow-up ─────────────────────────────────────────────────────────────
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

    # ── Medication count ──────────────────────────────────────────────────────
    meds = 0
    m = re.search(r'(\d+)\s*(?:medications?|meds?|drugs?|prescriptions?)', t)
    if m:
        meds = int(m.group(1))

    # ── Comorbidity count ─────────────────────────────────────────────────────
    comorbidities = 0
    m = re.search(r'(\d+)\s*(?:comorbidities|comorbidity|co-morbidities|conditions?)', t)
    if m:
        comorbidities = int(m.group(1))

    # ── Discharge destination ─────────────────────────────────────────────────
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
    """Normalize role labels into one of the supported target user groups."""
    if not role:
        return _DEFAULT_ROLE
    normalized = str(role).strip().lower()
    return _ROLE_LABELS.get(normalized, _DEFAULT_ROLE)


def _call_risk_scorer_tool(patient: dict) -> dict:
    """Call the risk scoring tool wrapper and parse its JSON response."""
    patient_json = json.dumps(patient, default=str)

    # Call the underlying tool function directly outside agent loops.
    raw = risk_scorer_tool.func(patient_json)
    result = json.loads(raw)
    if "error" in result:
        raise ValueError(result["error"])
    return result


def _extract_first_json_object(text: str) -> dict:
    """Extract and parse the first JSON object in a model response string."""
    start = text.find("{")
    end = text.rfind("}")
    if start == -1 or end == -1 or end <= start:
        raise ValueError("No JSON object found in LLM response")
    return json.loads(text[start : end + 1])


def _fallback_actions(risk: dict, role: str) -> list[str]:
    """Generate non-clinical preventive actions when LLM output is unavailable."""
    factors = risk.get("contributing_factors", [])
    level = risk.get("risk_level", "Low")
    actions = []

    if "No follow-up scheduled" in factors:
        actions.append("Schedule follow-up within 7 days and confirm patient receives appointment details.")
    if "Prior admissions ≥ 2 in last 6 months" in factors:
        actions.append("Arrange proactive outreach calls in the first 72 hours after discharge.")
    if "Length of stay > 7 days" in factors:
        actions.append("Create a discharge transition checklist and verify completion before handoff.")
    if "Medication count ≥ 8" in factors:
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
        actions.insert(0, "⚠️ ESCALATE: Immediate care coordinator review recommended.")

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
    """Render a strict assessment response format with mandatory sections."""
    badge = {
        "High": "🔴 HIGH",
        "Medium": "🟡 MEDIUM",
        "Low": "🟢 LOW",
    }.get(risk.get("risk_level", "Low"), "🟢 LOW")

    factors = risk.get("contributing_factors") or ["No major risk factors identified by scoring rules."]
    top_factors = factors[:3]

    if risk.get("risk_level") == "High" and not any("ESCALATE" in a for a in actions):
        actions = ["⚠️ ESCALATE: Immediate care coordinator review recommended."] + actions

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
        lines.append(f"• {factor}")

    lines.extend(["", "RECOMMENDED PREVENTIVE ACTIONS:"])
    for action in actions[:4]:
        lines.append(f"• {action}")

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
    """Create a tightly constrained prompt for JSON reasoning output."""
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
        "- If risk_level is High, include this exact action: \"⚠️ ESCALATE: Immediate care coordinator review recommended.\"\n"
        "- Tailor wording for this target user role: "
        f"{role}.\n\n"
        f"Original request: {query or 'N/A'}\n\n"
        f"Patient data:\n{json.dumps(patient, indent=2)}\n\n"
        f"Risk assessment:\n{json.dumps(risk, indent=2)}"
    )


def _llm_reasoning_payload(patient: dict, risk: dict, role: str, query: str) -> dict:
    """Ask the local LLM for explanatory reasoning and preventive actions."""
    llm = ChatOllama(
        model="llama3.2:1b",
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
    patient_data: Optional[dict] = None,
    allow_llm: bool = True,
) -> dict[str, Any]:
    """Run a unified interactive assessment with mandatory output sections.

    This is the primary path for interactive UI actions (Assess and Chat).
    It always calls the risk scoring tool and then attempts LLM reasoning.
    If LLM is unavailable, it falls back to deterministic interpretation.
    """
    role_name = _normalize_role(role)

    patient: Optional[dict] = None
    source_query = (query or "").strip()

    if patient_data is not None:
        patient = dict(patient_data)
    elif patient_id:
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
        pid_match = re.search(r"\bP\d{3}\b", source_query.upper())
        if pid_match:
            patient = load_patient_by_id(pid_match.group())
            if patient is None:
                return {
                    "error": f"Patient '{pid_match.group()}' not found.",
                    "response_text": (
                        f"READMISSION RISK ASSESSMENT\n"
                        f"Target User: {role_name}\n"
                        f"Patient ID: {pid_match.group()}\n\n"
                        f"Unable to locate this patient in the local dataset.\n\n"
                        f"{_SAFETY_DISCLAIMER}"
                    ),
                    "patient_data": None,
                    "risk_assessment": None,
                    "used_llm_reasoning": False,
                    "used_fallback": True,
                    "role": role_name,
                }
        elif source_query:
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


def run_agent_conversational(text: str) -> dict:
    """Score a patient described in free text without going through the LLM.

    Extracts patient attributes via regex, then runs the deterministic scorer.
    This is the primary path for Mode A (Conversational Input) to avoid
    agent iteration timeouts on llama3:8b.

    Args:
        text (str): Free-text patient description.

    Returns:
        dict: {
            "patient_data": dict,
            "risk_assessment": dict,
            "parsed_fields": dict,   ← shows what was extracted
        }
    """
    patient = parse_conversational_input(text)
    assessment = compute_risk_score(patient)
    return {
        "patient_data": patient,
        "risk_assessment": assessment,
        "parsed_fields": patient,
    }


# ── Agent construction (used for ID-based LLM narrative) ──────────────────────
def _build_agent() -> AgentExecutor:
    """Construct and return the LangChain AgentExecutor.

    Uses a ReAct-style agent with the Ollama llama3:8b model.
    Used only when a patient_id is available for lookup.

    Returns:
        AgentExecutor: Fully configured agent with tools and prompt.
    """
    system_text = _load_system_prompt()

    react_template = (
        system_text
        + """

You have access to the following tools:

{tools}

Use the following format EXACTLY — do not deviate:

Question: the input question you must answer
Thought: you should always think about what to do
Action: the action to take, should be one of [{tool_names}]
Action Input: the input to the action (a plain string or JSON)
Observation: the result of the action
Thought: I now know the final answer
Final Answer: the final answer to the original input question

Rules:
- Always call patient_lookup_tool first with the patient ID.
- Then call risk_scorer_tool with the JSON patient data.
- Then write the Final Answer using the structured assessment format.
- Do NOT call tools more than once each.
- Do NOT guess or invent a risk score — always use the tool result.

Begin!

Question: {input}
Thought:{agent_scratchpad}"""
    )

    prompt = PromptTemplate.from_template(react_template)

    llm = ChatOllama(
        model="llama3.2:3b",
        temperature=0.1,
        num_predict=1500,
    )

    tools = [patient_lookup_tool, risk_scorer_tool]
    agent = create_react_agent(llm=llm, tools=tools, prompt=prompt)

    executor = AgentExecutor(
        agent=agent,
        tools=tools,
        verbose=True,
        handle_parsing_errors=True,
        max_iterations=8,
        return_intermediate_steps=True,
    )
    return executor


# ── Cached executor ────────────────────────────────────────────────────────────
_agent_executor: Union[AgentExecutor, None] = None


def _get_executor() -> AgentExecutor:
    """Return a cached AgentExecutor (lazy initialisation).

    Returns:
        AgentExecutor: The singleton agent executor instance.
    """
    global _agent_executor
    if _agent_executor is None:
        _agent_executor = _build_agent()
    return _agent_executor


def run_agent(query: str) -> str:
    """Run the readmission risk agent for a query that contains a patient ID.

    Intended for patient-ID-based queries only. For free-text patient
    descriptions use run_agent_conversational() instead to avoid iteration
    timeouts with llama3:8b.

    Args:
        query (str): Natural language input referencing a patient ID.

    Returns:
        str: Structured risk assessment text with disclaimer.
    """
    if not query or not query.strip():
        return (
            "⚠️ No input provided. Please enter a patient ID or patient details "
            "to generate a readmission risk assessment."
        )

    try:
        executor = _get_executor()
        result = executor.invoke({"input": query})
        output = result.get("output", "")

        disclaimer = (
            "\n\n⚠️ DISCLAIMER: This is a decision-support tool. "
            "Always consult a licensed clinician before acting on this output."
        )
        if "disclaimer" not in output.lower() and "advisory" not in output.lower():
            output += disclaimer

        return output

    except Exception as exc:
        err = str(exc)
        if "connection" in err.lower() or "refused" in err.lower():
            return (
                "❌ Cannot connect to Ollama.\n\n"
                "Please ensure Ollama is running:\n"
                "  1. Open a terminal\n"
                "  2. Run: `ollama serve`\n"
                "  3. In another terminal, verify: `ollama list`\n"
                "  4. Confirm llama3:8b is listed, then retry."
            )
        return f"❌ Agent error: {err}"


def run_agent_direct(patient_id: str) -> dict:
    """Directly score a patient by ID without going through the LLM.

    Fast, deterministic path used for CSV lookup mode and batch validation.

    Args:
        patient_id (str): Patient identifier string, e.g. 'P007'.

    Returns:
        dict: {
            "patient_id": str,
            "patient_data": dict or None,
            "risk_assessment": dict or None,
            "error": str or None
        }
    """
    record = load_patient_by_id(patient_id)
    if record is None:
        return {
            "patient_id": patient_id,
            "patient_data": None,
            "risk_assessment": None,
            "error": "Patient not found",
        }

    assessment = compute_risk_score(record)
    return {
        "patient_id": patient_id,
        "patient_data": record,
        "risk_assessment": assessment,
        "error": None,
    }
