"""
agent.py — LangChain AI Agent for Hospital Readmission Risk Assessment.

Uses Ollama (llama3:8b) as the local LLM with two tools:
  1. patient_lookup_tool  — retrieves patient data from CSV
  2. risk_scorer_tool     — computes rule-based risk score

All inference runs fully offline. No cloud API calls are made.

Conversational input (free text without a patient ID) is handled via a
deterministic extraction path — no LLM iteration loop required.
"""

import json
import os
import re
from typing import Optional, Union

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
