"""
Minimal Hospital Discharge Planning Hub
Real-world Readmission Risk Assessment Tool
Runs Fully Offline · Decision-Support Only
"""

import streamlit as st
import pandas as pd
from datetime import datetime
import re

from agent import run_agent_direct, run_agent_conversational
from tools.risk_scorer import compute_risk_score
from utils.data_loader import load_all_patients, list_patient_ids


# -----------------------------------------------------------------------------
# Page Configuration
# -----------------------------------------------------------------------------
st.set_page_config(
    page_title="Discharge Planning Hub",
    page_icon="🏥",
    layout="wide"
)

# -----------------------------------------------------------------------------
# Minimal Styling
# -----------------------------------------------------------------------------
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

# -----------------------------------------------------------------------------
# Cached Data
# -----------------------------------------------------------------------------
@st.cache_data
def get_all_patients():
    df = load_all_patients()
    rows = []
    for _, r in df.iterrows():
        risk = compute_risk_score(r.to_dict())
        rows.append({
            **r.to_dict(),
            "risk_level": risk["risk_level"],
            "risk_score": risk["score"]
        })
    return pd.DataFrame(rows)

@st.cache_data
def get_patient_ids():
    return list_patient_ids()

# -----------------------------------------------------------------------------
# Header
# -----------------------------------------------------------------------------
st.title("🏥 Discharge Planning Hub")
st.caption(
    "30-Day Readmission Risk Assessment · Decision-Support Tool · Runs Offline"
)
st.divider()

# -----------------------------------------------------------------------------
# Sidebar
# -----------------------------------------------------------------------------
with st.sidebar:
    st.header("⚙️ Controls")

    if st.button("▶ Run Demo (P007)"):
        st.session_state["demo"] = "P007"

    if st.button("🗑 Clear Session"):
        st.session_state.clear()
        st.rerun()

    st.divider()
    st.caption("Decision-Support Only")
    st.caption("Synthetic Data · Offline Mode")

# -----------------------------------------------------------------------------
# Tabs
# -----------------------------------------------------------------------------
tab_dashboard, tab_assess, tab_chat, tab_records = st.tabs([
    "📊 Dashboard",
    "🧑‍⚕️ Assess Patient",
    "💬 AI Assistant",
    "📋 Records"
])

# =============================================================================
# TAB 1 — DASHBOARD
# =============================================================================
with tab_dashboard:
    st.subheader("Risk Overview")

    df = get_all_patients()

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

# =============================================================================
# TAB 2 — ASSESS PATIENT
# =============================================================================
with tab_assess:
    st.subheader("Assess Readmission Risk")

    mode = st.radio(
        "Select Mode",
        ["Existing Patient", "New Patient"],
        horizontal=True
    )

    if mode == "Existing Patient":
        ids = get_patient_ids()
        default_pid = st.session_state.get("demo", ids[0])

        pid = st.selectbox("Patient ID", ids, index=ids.index(default_pid))

        if st.button("Assess Patient"):
            result = run_agent_direct(pid)

            if result["error"]:
                st.error("Patient not found.")
            else:
                risk = result["risk_assessment"]
                pdata = result["patient_data"]

                st.success(f"Risk Level: {risk['risk_level']}")
                st.metric("Risk Score", f"{risk['score']}/10")

                st.markdown("### Contributing Factors")
                for factor in risk["contributing_factors"]:
                    st.write(f"• {factor}")

                st.markdown("### Patient Summary")
                st.json(pdata)

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

            risk = compute_risk_score(patient)

            st.success(f"Risk Level: {risk['risk_level']}")
            st.metric("Risk Score", f"{risk['score']}/10")

            st.markdown("### Contributing Factors")
            for factor in risk["contributing_factors"]:
                st.write(f"• {factor}")

# =============================================================================
# TAB 3 — AI ASSISTANT
# =============================================================================
with tab_chat:
    st.subheader("AI Discharge Assistant")

    user_input = st.chat_input(
        "Enter a patient ID or describe a patient..."
    )

    if "chat_history" not in st.session_state:
        st.session_state.chat_history = []

    if user_input:
        st.session_state.chat_history.append(("user", user_input))

        pid_match = re.search(r"\bP\d{3}\b", user_input.upper())

        if pid_match:
            result = run_agent_direct(pid_match.group())
        else:
            result = run_agent_conversational(user_input)

        risk = result["risk_assessment"]

        reply = (
            f"**Risk Level:** {risk['risk_level']}  \n"
            f"**Risk Score:** {risk['score']}/10"
        )

        st.session_state.chat_history.append(("assistant", reply))

    for role, message in st.session_state.chat_history:
        with st.chat_message(role):
            st.markdown(message)

# =============================================================================
# TAB 4 — PATIENT RECORDS
# =============================================================================
with tab_records:
    st.subheader("Patient Records")

    df = get_all_patients()

    risk_filter = st.multiselect(
        "Filter by Risk Level",
        ["High", "Medium", "Low"]
    )

    if risk_filter:
        df = df[df["risk_level"].isin(risk_filter)]

    st.dataframe(df, use_container_width=True, hide_index=True)

    csv = df.to_csv(index=False).encode("utf-8")

    st.download_button(
        "⬇ Download CSV",
        data=csv,
        file_name="patient_records.csv",
        mime="text/csv"
    )

# -----------------------------------------------------------------------------
# Footer
# -----------------------------------------------------------------------------
st.divider()
st.caption(
    f"Generated on {datetime.now().strftime('%d %B %Y')} · "
    "Hospital Discharge Planning Hub · Offline AI Tool"
)