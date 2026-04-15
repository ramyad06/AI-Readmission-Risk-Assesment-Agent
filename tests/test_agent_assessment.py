"""
test_agent_assessment.py — Tests for unified interactive assessment output contract.

These tests run in deterministic fallback mode (allow_llm=False), so they do not
require a running Ollama instance.
"""

import os
import sys

# Allow import from project root
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from agent import run_assessment_with_reasoning


def _sample_new_patient(high_risk: bool = False) -> dict:
    if high_risk:
        return {
            "patient_id": "NEW-HIGH",
            "age": 83,
            "primary_condition": "Heart Failure",
            "length_of_stay_days": 12,
            "prior_admissions_6mo": 3,
            "discharge_destination": "Home",
            "follow_up_scheduled": "No",
            "medication_count": 10,
            "comorbidity_count": 4,
        }
    return {
        "patient_id": "NEW-LOW",
        "age": 45,
        "primary_condition": "Diabetes",
        "length_of_stay_days": 3,
        "prior_admissions_6mo": 0,
        "discharge_destination": "Home+Services",
        "follow_up_scheduled": "Yes",
        "medication_count": 2,
        "comorbidity_count": 1,
    }


def test_assessment_response_has_required_sections():
    result = run_assessment_with_reasoning(
        role="Care Coordinators",
        patient_data=_sample_new_patient(high_risk=False),
        allow_llm=False,
    )

    assert result["error"] is None
    text = result["response_text"]

    assert "READMISSION RISK ASSESSMENT" in text
    assert "Readmission Risk Level:" in text
    assert "TOP CONTRIBUTING FACTORS:" in text
    assert "RECOMMENDED PREVENTIVE ACTIONS:" in text
    assert "⚠️ SAFETY DISCLAIMER:" in text


def test_high_risk_response_includes_escalation():
    result = run_assessment_with_reasoning(
        role="Nursing Staff",
        patient_data=_sample_new_patient(high_risk=True),
        allow_llm=False,
    )

    assert result["error"] is None
    assert result["risk_assessment"]["risk_level"] == "High"
    assert "⚠️ ESCALATE: Immediate care coordinator review recommended." in result["response_text"]


def test_missing_patient_id_returns_graceful_error():
    result = run_assessment_with_reasoning(
        query="Assess patient P999",
        role="Case Managers",
        allow_llm=False,
    )

    assert result["error"] is not None
    assert "not found" in result["error"].lower()
    assert "⚠️ SAFETY DISCLAIMER:" in result["response_text"]


def test_conversational_input_is_supported_in_unified_flow():
    result = run_assessment_with_reasoning(
        query="72 year old with COPD, 3 prior admissions, no follow-up booked",
        role="Hospital Operations Team",
        allow_llm=False,
    )

    assert result["error"] is None
    assert result["patient_data"] is not None
    assert result["risk_assessment"] is not None
    assert result["used_llm_reasoning"] is False
    assert result["used_fallback"] is True
