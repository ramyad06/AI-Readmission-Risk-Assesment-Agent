import streamlit as st
import pandas as pd

from agent import run_assessment_with_reasoning
from tools.risk_scorer import compute_risk_score
from utils.data_loader import load_all_patients, list_patient_ids

st.set_page_config(
    page_title="Discharge Planning Hub",
    layout="wide"
)

st.markdown("""
<style>
.block-container {
    padding-top: 2rem;
    padding-bottom: 1rem;
    padding-left: 2rem;
    padding-right: 2rem;
}
.metric-card {
    background-color: #ffffff;
    border: 1px solid #e5e7eb;
    padding: 15px;
    border-radius: 10px;
}
</style>
""", unsafe_allow_html=True)

def _parse_assessment_response(text: str) -> dict:
    parsed = {
        "target_user": "",
        "patient_id": "",
        "risk_level": "",
        "risk_score": "",
        "reasoning_summary": "",
        "factors": [],
        "actions": [],
        "warning": "",
        "disclaimer": "",
    }

    section = None
    for raw_line in str(text or "").splitlines():
        line = raw_line.strip()
        if not line:
            continue

        if line.startswith("Target User:"):
            parsed["target_user"] = line.split(":", 1)[1].strip()
            section = None
            continue
        if line.startswith("Patient ID:"):
            parsed["patient_id"] = line.split(":", 1)[1].strip()
            section = None
            continue
        if line.startswith("Readmission Risk Level:"):
            parsed["risk_level"] = line.split(":", 1)[1].strip()
            section = None
            continue
        if line.startswith("Risk Score:"):
            parsed["risk_score"] = line.split(":", 1)[1].strip()
            section = None
            continue

        if line == "REASONING SUMMARY:":
            section = "summary"
            continue
        if line == "TOP CONTRIBUTING FACTORS:":
            section = "factors"
            continue
        if line == "RECOMMENDED PREVENTIVE ACTIONS:":
            section = "actions"
            continue

        if line.startswith("LLM reasoning unavailable"):
            parsed["warning"] = line
            section = None
            continue
        if line.startswith("SAFETY DISCLAIMER:"):
            parsed["disclaimer"] = line
            section = None
            continue

        if line.startswith("- "):
            item = line[2:].strip()
            if section == "factors":
                parsed["factors"].append(item)
            elif section == "actions":
                parsed["actions"].append(item)
            continue

        if section == "summary":
            if parsed["reasoning_summary"]:
                parsed["reasoning_summary"] += " "
            parsed["reasoning_summary"] += line

    return parsed


def _format_assessment_markdown(text: str) -> str:
    p = _parse_assessment_response(text)

    lines = [
        "### READMISSION RISK ASSESSMENT",
        "",
        f"**Target User:** {p['target_user'] or 'N/A'}",
        f"**Patient ID:** {p['patient_id'] or 'N/A'}",
        f"**Readmission Risk Level:** {p['risk_level'] or 'N/A'}",
        f"**Risk Score:** {p['risk_score'] or 'N/A'}",
        "",
        "#### REASONING SUMMARY",
        p["reasoning_summary"] or "No reasoning summary available.",
        "",
        "#### TOP CONTRIBUTING FACTORS",
    ]

    if p["factors"]:
        for factor in p["factors"]:
            lines.append(f"- {factor}")
    else:
        lines.append("- No contributing factors listed.")

    lines.extend(["", "#### RECOMMENDED PREVENTIVE ACTIONS"])
    if p["actions"]:
        for action in p["actions"]:
            lines.append(f"- {action}")
    else:
        lines.append("- No preventive actions listed.")

    if p["warning"]:
        lines.extend(["", f"> {p['warning']}"])

    if p["disclaimer"]:
        lines.extend(["", f"> {p['disclaimer']}"])

    return "\n".join(lines)

st.title("Discharge Planning Hub")
st.caption(
    "30-Day Readmission Risk Assessment - Decision-Support Tool"
)
st.divider()

st.subheader("Target User")
target_role = st.selectbox(
    "Select user group",
    [
        "Care Coordinators",
        "Hospital Operations Team",
        "Case Managers",
        "Nursing Staff",
    ],
    index=0,
    help="Interactive assessments tailor non-clinical actions for this user group.",
)

if st.button("Clear Session"):
    st.session_state.clear()
    st.rerun()

st.divider()

tab_dashboard, tab_assess, tab_chat, tab_records = st.tabs([
    "Dashboard",
    "Assess Patient",
    "AI Assistant",
    "Records"
])

with tab_dashboard:
    st.subheader("Risk Overview")
    st.caption("Operational overview only: deterministic scoring is used in this tab.")

    df = load_all_patients()

    col1, col2, col3, col4 = st.columns(4)
    col1.metric("High Risk", (df["risk_level"] == "High").sum())
    col2.metric("Medium Risk", (df["risk_level"] == "Medium").sum())
    col3.metric("Low Risk", (df["risk_level"] == "Low").sum())
    col4.metric("Average Score", f"{df['risk_score'].mean():.1f}/10")

    st.markdown("### Patient Risk Table")

    st.dataframe(
        df[[
            "patient_id", "age", "primary_condition",
            "risk_level", "risk_score",
            "length_of_stay_days", "follow_up_scheduled"
        ]],
        use_container_width=True,
        hide_index=True
    )

with tab_assess:
    st.subheader("Assess Readmission Risk")
    st.caption("Interactive assessments use LLM reasoning plus rule-based scoring.")

    mode = st.radio(
        "Select Mode",
        ["Existing Patient", "New Patient"],
        horizontal=True
    )

    if mode == "Existing Patient":
        ids = list_patient_ids()
        pid = st.selectbox("Patient ID", ids, index=0)

        if st.button("Assess Patient"):
            result = run_assessment_with_reasoning(
                query=f"Assess patient {pid} for 30-day readmission risk.",
                role=target_role,
                patient_id=pid,
            )

            if result["error"]:
                st.error(result["error"])
            else:
                st.markdown(_format_assessment_markdown(result["response_text"]))
                with st.expander("Patient Summary", expanded=False):
                    st.json(result["patient_data"])

    else:
        with st.form("new_patient"):
            col1, col2, col3 = st.columns(3)

            with col1:
                age = st.number_input("Age", 18, 110, 65)
                los = st.number_input("Length of Stay", 0, 30, 5)
                meds = st.number_input("Medication Count", 0, 20, 5)

            with col2:
                condition = st.selectbox(
                    "Primary Condition",
                    ["Heart Failure", "COPD", "Diabetes", "Pneumonia", "Hip Fracture", "Other"]
                )
                prior = st.number_input("Prior Admissions (6mo)", 0, 10, 0)
                comorb = st.number_input("Comorbidities", 0, 10, 0)

            with col3:
                discharge = st.selectbox(
                    "Discharge Destination",
                    ["Home", "Home+Services", "SNF", "Rehab"]
                )
                follow_up = st.selectbox("Follow-up Scheduled", ["Yes", "No"])

            submitted = st.form_submit_button("Calculate Risk")

        if submitted:
            patient = {
                "patient_id": "NEW",
                "age": age,
                "primary_condition": condition,
                "length_of_stay_days": los,
                "prior_admissions_6mo": prior,
                "discharge_destination": discharge,
                "follow_up_scheduled": follow_up,
                "medication_count": meds,
                "comorbidity_count": comorb,
            }

            result = run_assessment_with_reasoning(
                query=(
                    "Assess this new patient profile for 30-day readmission risk and "
                    "provide non-clinical preventive actions."
                ),
                role=target_role,
                patient_data=patient,
            )

            if result["error"]:
                st.error(result["error"])
            st.markdown(_format_assessment_markdown(result["response_text"]))
            with st.expander("Patient Summary", expanded=False):
                st.json(result["patient_data"])
 
with tab_chat:
    st.subheader("AI Discharge Assistant")
    st.caption("Chat responses use LLM reasoning with mandatory safety and action sections.")

    user_input = st.chat_input(
        "Enter a patient ID or describe a patient..."
    )

    if "chat_history" not in st.session_state:
        st.session_state.chat_history = []

    if user_input:
        st.session_state.chat_history.append(("user", user_input))

        result = run_assessment_with_reasoning(
            query=user_input,
            role=target_role,
        )

        if result["error"]:
            reply = (
                f"{_format_assessment_markdown(result['response_text'])}\n\n"
                f"**Error:** {result['error']}"
            )
        else:
            reply = _format_assessment_markdown(result["response_text"])

        st.session_state.chat_history.append(("assistant", reply))

    for role, message in st.session_state.chat_history:
        with st.chat_message(role):
            st.markdown(message)

with tab_records:
    st.subheader("Patient Records")
    st.caption("Operational records view uses deterministic scoring for fast filtering/export.")

    df = load_all_patients()

    risk_filter = st.multiselect(
        "Filter by Risk Level",
        ["High", "Medium", "Low"]
    )

    if risk_filter:
        df = df[df["risk_level"].isin(risk_filter)]

    st.dataframe(df, use_container_width=True, hide_index=True)

    csv = df.to_csv(index=False).encode("utf-8")

    st.download_button(
        "Download CSV",
        data=csv,
        file_name="patient_records.csv",
        mime="text/csv"
    )

st.divider()
st.caption("Decision-Support Only")
st.caption("Synthetic Data - Offline Mode")