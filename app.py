"""
app.py — Hospital Discharge Planning Hub
Real-world Readmission Risk Assessment Tool for Care Coordinators

Use case: Used by discharge coordinators during morning rounds to flag
          high-risk patients before they leave the hospital, and to plan
          targeted post-discharge interventions.

Runs fully offline · No real patient data · Decision-support only.
"""

import sys, os, re, json
from datetime import datetime, date
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

import streamlit as st
import pandas as pd

from agent import run_agent_direct, run_agent_conversational
from tools.risk_scorer import compute_risk_score
from utils.data_loader import load_all_patients, list_patient_ids

# ══════════════════════════════════════════════════════════════════════════════
# Page config
# ══════════════════════════════════════════════════════════════════════════════
st.set_page_config(
    page_title="Discharge Planning Hub · Readmission Risk",
    page_icon="🏥",
    layout="wide",
    initial_sidebar_state="expanded",
)

# ══════════════════════════════════════════════════════════════════════════════
# CSS — Clinical dark theme
# ══════════════════════════════════════════════════════════════════════════════
st.markdown("""
<style>
@import url('https://fonts.googleapis.com/css2?family=Inter:wght@300;400;500;600;700;800&display=swap');
*, *::before, *::after { box-sizing: border-box; margin:0; }

html, body, [class*="css"] { font-family:'Inter',sans-serif; }
.stApp          { background:#0d1117; color:#c9d1d9; }
.main > div     { padding-top: 1rem !important; }

/* Sidebar */
section[data-testid="stSidebar"]            { background:#0d1117 !important; border-right:1px solid #21262d !important; }
section[data-testid="stSidebar"] *          { color:#8b949e !important; }
section[data-testid="stSidebar"] .stButton > button { background:#161b22 !important; color:#c9d1d9 !important;
    border:1px solid #30363d !important; border-radius:8px !important; width:100%; font-size:0.83rem; }
section[data-testid="stSidebar"] .stButton > button:hover { border-color:#388bfd !important; }

/* Tabs */
.stTabs [data-baseweb="tab-list"] { gap:3px; background:#161b22; border-radius:12px; padding:4px; border:1px solid #21262d; margin-bottom:20px; }
.stTabs [data-baseweb="tab"]       { border-radius:8px; padding:10px 22px; color:#8b949e; font-weight:500; font-size:0.88rem; background:transparent; border:none; }
.stTabs [aria-selected="true"]     { background:linear-gradient(135deg,#1f6feb,#388bfd) !important; color:#fff !important; font-weight:600 !important; }

/* Global buttons */
.stButton > button {
    background:linear-gradient(135deg,#1f6feb,#388bfd); color:#fff; border:none;
    border-radius:8px; font-weight:600; font-size:0.88rem; padding:9px 18px;
    transition:all 0.18s ease; width:100%;
}
.stButton > button:hover { transform:translateY(-1px); box-shadow:0 4px 14px rgba(56,139,253,0.35); }

/* Inputs */
.stSelectbox > div > div,
.stTextInput > div > div > input,
.stNumberInput > div > div > input,
.stTextArea textarea {
    background:#161b22 !important; border:1px solid #30363d !important;
    border-radius:8px !important; color:#c9d1d9 !important; font-size:0.88rem !important;
}
.stSelectbox > div > div:focus-within,
.stTextArea textarea:focus,
.stTextInput > div > div > input:focus { border-color:#388bfd !important; box-shadow:0 0 0 3px rgba(56,139,253,0.12) !important; }
.stTextArea textarea::placeholder,
.stTextInput > div > div > input::placeholder { color:#484f58 !important; }
label { color:#8b949e !important; font-size:0.8rem !important; font-weight:500 !important; }

/* Metrics */
[data-testid="stMetricValue"]  { color:#e6edf3 !important; font-size:1.6rem !important; font-weight:800 !important; }
[data-testid="stMetricLabel"]  { color:#8b949e !important; font-size:0.73rem !important; }
[data-testid="stMetricDelta"]  { font-size:0.75rem !important; }

/* DataFrames */
.stDataFrame thead tr th  { background:#161b22 !important; color:#8b949e !important; border-bottom:1px solid #21262d !important; }
.stDataFrame { border-radius:10px; overflow:hidden; }

/* Expander */
.streamlit-expanderHeader { background:#161b22 !important; border:1px solid #21262d !important;
    border-radius:8px !important; color:#8b949e !important; font-size:0.85rem !important; }

/* Slider */
.stSlider [data-baseweb="slider"] { color:#388bfd; }

/* Progress */
.stProgress > div > div { background:#388bfd !important; border-radius:4px; }

hr { border-color:#21262d !important; margin:12px 0 !important; }

/* ── Custom components ─────────────────────────────────── */

.page-header {
    background:linear-gradient(135deg,#0d1117 0%,#161b22 100%);
    border:1px solid #21262d; border-radius:14px;
    padding:22px 28px; margin-bottom:22px; position:relative; overflow:hidden;
}
.page-header::before {
    content:''; position:absolute; top:-40px; right:-40px;
    width:160px; height:160px;
    background:radial-gradient(circle,rgba(56,139,253,0.1) 0%,transparent 70%);
    border-radius:50%;
}
.ph-badge   { display:inline-block; background:rgba(63,185,80,0.12); border:1px solid rgba(63,185,80,0.3);
    color:#3fb950; font-size:0.68rem; font-weight:700; padding:2px 10px;
    border-radius:20px; letter-spacing:0.08em; margin-bottom:8px; }
.ph-title   { font-size:1.55rem; font-weight:800; color:#e6edf3; letter-spacing:-0.02em; }
.ph-sub     { font-size:0.85rem; color:#8b949e; margin-top:4px; }
.ph-meta    { font-size:0.75rem; color:#484f58; margin-top:8px; }

/* KPI cards */
.kpi-grid { display:grid; grid-template-columns:repeat(4,1fr); gap:12px; margin-bottom:22px; }
.kpi-card {
    background:#161b22; border:1px solid #21262d; border-radius:12px;
    padding:16px 18px; cursor:pointer; transition:border-color 0.18s, transform 0.15s;
}
.kpi-card:hover { border-color:#30363d; transform:translateY(-1px); }
.kpi-num  { font-size:2rem; font-weight:800; color:#e6edf3; line-height:1; }
.kpi-lbl  { font-size:0.72rem; color:#8b949e; margin-top:4px; font-weight:500; text-transform:uppercase; letter-spacing:0.06em; }
.kpi-sub  { font-size:0.75rem; color:#484f58; margin-top:2px; }

/* Risk row table */
.risk-row {
    display:grid; grid-template-columns:80px 50px 150px 90px 80px 80px 80px 1fr;
    align-items:center; gap:12px; padding:10px 14px;
    border-bottom:1px solid #21262d; font-size:0.83rem; color:#c9d1d9;
    transition:background 0.12s;
}
.risk-row:hover { background:rgba(255,255,255,0.02); }
.risk-row-header { color:#8b949e !important; font-weight:600; font-size:0.72rem;
    text-transform:uppercase; letter-spacing:0.07em; background:#161b22;
    border-radius:10px 10px 0 0; }

/* Risk badges */
.rb { display:inline-block; font-size:0.72rem; font-weight:700; padding:3px 10px;
    border-radius:20px; letter-spacing:0.05em; }
.rb-H { background:rgba(248,81,73,0.15); color:#f85149; border:1px solid rgba(248,81,73,0.3); }
.rb-M { background:rgba(210,153,34,0.15); color:#e3b341; border:1px solid rgba(210,153,34,0.3); }
.rb-L { background:rgba(63,185,80,0.15);  color:#3fb950; border:1px solid rgba(63,185,80,0.3); }

/* Score bar in table */
.mini-bar-wrap { display:flex; align-items:center; gap:8px; }
.mini-bar-bg   { flex:1; background:#21262d; border-radius:4px; height:6px; overflow:hidden; }
.mini-bar-fill { height:100%; border-radius:4px; }

/* Full assessment card */
.acard {
    background:#161b22; border:1px solid #21262d;
    border-radius:14px; padding:24px 26px; margin-bottom:16px;
}
.acard-section-title {
    font-size:0.68rem; font-weight:700; color:#8b949e; text-transform:uppercase;
    letter-spacing:0.1em; margin:18px 0 10px; display:flex; align-items:center; gap:8px;
}
.acard-section-title::after { content:''; flex:1; height:1px; background:#21262d; }

/* SVG gauge */
.gauge-wrap { display:flex; flex-direction:column; align-items:center; gap:6px; }
.gauge-label { font-size:0.68rem; color:#8b949e; font-weight:600;
    text-transform:uppercase; letter-spacing:0.08em; }

/* Risk big badge */
.big-badge { display:inline-flex; align-items:center; gap:10px; font-size:1.05rem;
    font-weight:700; padding:11px 24px; border-radius:50px; letter-spacing:0.04em; }
.bb-H { background:linear-gradient(135deg,#da3633,#f85149); color:#fff;
    border:1px solid rgba(248,81,73,0.5); box-shadow:0 0 24px rgba(248,81,73,0.3);
    animation:pulse-h 2.5s ease-in-out infinite; }
.bb-M { background:linear-gradient(135deg,#9e6a03,#d29922); color:#fff;
    border:1px solid rgba(210,153,34,0.5); box-shadow:0 0 18px rgba(210,153,34,0.25); }
.bb-L { background:linear-gradient(135deg,#196c2e,#2ea043); color:#fff;
    border:1px solid rgba(63,185,80,0.5); box-shadow:0 0 18px rgba(63,185,80,0.2); }
@keyframes pulse-h { 0%,100%{box-shadow:0 0 24px rgba(248,81,73,0.3);}
    50%{box-shadow:0 0 40px rgba(248,81,73,0.6);} }

/* Score progress bar */
.score-bar-outer { background:#21262d; border-radius:6px; height:10px; overflow:hidden; margin:8px 0; }
.score-bar-inner { height:100%; border-radius:6px; }

/* Factor chips */
.chip { display:inline-block; background:rgba(56,139,253,0.1); border:1px solid rgba(56,139,253,0.25);
    color:#79c0ff; font-size:0.78rem; padding:4px 12px; border-radius:20px; margin:3px 3px; font-weight:500; }
.chip-ok { background:rgba(63,185,80,0.1); border-color:rgba(63,185,80,0.25); color:#56d364; }

/* Action items */
.action { display:flex; align-items:flex-start; gap:10px; padding:10px 14px;
    background:#0d1117; border:1px solid #21262d; border-radius:8px;
    margin-bottom:8px; font-size:0.85rem; color:#c9d1d9; }
.action-esc { border-color:rgba(248,81,73,0.4) !important;
    background:rgba(248,81,73,0.06) !important; color:#ffa198 !important; font-weight:600; }

/* Disclaimer */
.disc { background:rgba(210,153,34,0.07); border:1px solid rgba(210,153,34,0.25);
    border-radius:10px; padding:12px 16px; margin-top:16px;
    font-size:0.8rem; color:#b08800; line-height:1.6; }
.disc-esc { background:rgba(248,81,73,0.07); border-color:rgba(248,81,73,0.3);
    color:#cf6679; margin-top:8px; }

/* Checklist */
.cl-item { display:flex; align-items:flex-start; gap:10px; padding:10px 14px;
    background:#0d1117; border:1px solid #21262d; border-radius:8px; margin-bottom:7px;
    font-size:0.85rem; color:#c9d1d9; }
.cl-done { border-color:rgba(63,185,80,0.3) !important;
    background:rgba(63,185,80,0.05) !important; color:#56d364 !important; text-decoration:line-through; }

/* Chat */
.bubble-u { display:flex; justify-content:flex-end; margin-bottom:12px; }
.bubble-u-inner { background:linear-gradient(135deg,#1f6feb,#388bfd); color:#fff;
    border-radius:16px 16px 4px 16px; padding:10px 16px; max-width:82%;
    font-size:0.87rem; line-height:1.5; }
.bubble-a { display:flex; gap:10px; margin-bottom:12px; align-items:flex-start; }
.bubble-a-av { width:30px; height:30px; background:linear-gradient(135deg,#238636,#2ea043);
    border-radius:50%; display:flex; align-items:center; justify-content:center;
    font-size:0.85rem; flex-shrink:0; }
.bubble-a-inner { background:#161b22; border:1px solid #21262d; color:#c9d1d9;
    border-radius:4px 16px 16px 16px; padding:10px 16px; max-width:82%;
    font-size:0.87rem; line-height:1.5; }

/* Patient info strip */
.pstrip { display:grid; grid-template-columns:repeat(4,1fr); gap:10px; margin-bottom:16px; }
.pcell { background:#0d1117; border:1px solid #21262d; border-radius:10px;
    padding:12px; text-align:center; }
.pcell-val { font-size:1.05rem; font-weight:700; color:#e6edf3; }
.pcell-lbl { font-size:0.67rem; color:#8b949e; margin-top:3px;
    text-transform:uppercase; letter-spacing:0.06em; }

/* Info banner */
.info-banner { background:rgba(56,139,253,0.07); border:1px solid rgba(56,139,253,0.2);
    border-radius:10px; padding:12px 16px; margin-bottom:14px;
    font-size:0.82rem; color:#58a6ff; }

/* Empty state */
.empty { text-align:center; padding:48px 24px; color:#484f58; }
.empty-icon  { font-size:2.8rem; margin-bottom:10px; }
.empty-title { font-size:1rem; font-weight:600; color:#6e7681; margin-bottom:6px; }
.empty-sub   { font-size:0.83rem; }

/* Report box */
.report-box { background:#0d1117; border:1px solid #21262d; border-radius:10px;
    padding:18px; font-family:monospace; font-size:0.8rem;
    color:#c9d1d9; white-space:pre-wrap; line-height:1.7; max-height:460px; overflow-y:auto; }

/* Sidebar nav */
.sb-logo { padding:12px 0 6px; border-bottom:1px solid #21262d; margin-bottom:14px; }
.sb-logo-title { font-size:1.1rem; font-weight:800; color:#e6edf3; }
.sb-logo-sub   { font-size:0.73rem; color:#484f58; margin-top:1px; }
.sb-section    { font-size:0.65rem; font-weight:700; color:#484f58;
    text-transform:uppercase; letter-spacing:0.1em; margin:16px 0 8px; }
.sb-stat { display:flex; justify-content:space-between; align-items:center;
    padding:8px 12px; background:#161b22; border:1px solid #21262d;
    border-radius:8px; margin-bottom:6px; }
.sb-stat-l { font-size:0.78rem; color:#8b949e; }
.sb-stat-v { font-size:0.9rem; font-weight:700; color:#e6edf3; }
</style>
""", unsafe_allow_html=True)


# ══════════════════════════════════════════════════════════════════════════════
# Session state
# ══════════════════════════════════════════════════════════════════════════════
def _init():
    defaults = {
        "chat_history":   [],       # [{role, content, assessment?, pid?, pdata?}]
        "dashboard_sel":  None,     # currently expanded patient on dashboard
        "assess_result":  None,     # result from Assess tab
        "checklist":      {},       # {patient_id: {item_idx: bool}}
        "report_text":    "",       # last generated report text
    }
    for k, v in defaults.items():
        if k not in st.session_state:
            st.session_state[k] = v
_init()


# ══════════════════════════════════════════════════════════════════════════════
# Cached data
# ══════════════════════════════════════════════════════════════════════════════
@st.cache_data(ttl=120)
def _all_patients():
    """Load all 40 patients and compute risk scores for every one."""
    df = load_all_patients()
    rows = []
    for _, r in df.iterrows():
        a = compute_risk_score(r.to_dict())
        rows.append({**r.to_dict(), **{
            "risk_level": a["risk_level"],
            "risk_score": a["score"],
            "factors":    a["contributing_factors"],
            "breakdown":  a["score_breakdown"],
        }})
    return pd.DataFrame(rows)

@st.cache_data(ttl=120)
def _get_ids():
    return list_patient_ids()


# ══════════════════════════════════════════════════════════════════════════════
# UI helpers
# ══════════════════════════════════════════════════════════════════════════════
_RISK_COLOR   = {"High": "#f85149", "Medium": "#e3b341", "Low": "#3fb950"}
_RISK_CLR_BG  = {"High": "rgba(248,81,73,0.1)", "Medium": "rgba(210,153,34,0.1)", "Low": "rgba(63,185,80,0.08)"}
_RISK_ICON    = {"High": "🔴", "Medium": "🟡", "Low": "🟢"}

_ACTIONS = {
    "High": [
        (True,  "🚨", "ESCALATE — Notify care coordinator & attending physician immediately"),
        (False, "📞", "Arrange post-discharge phone call within 24 hours"),
        (False, "📅", "Schedule in-person follow-up within 48 hours of discharge"),
        (False, "🏠", "Initiate home health referral before patient leaves"),
        (False, "💊", "Complete medication reconciliation and patient counselling"),
        (False, "📋", "Document high-risk flag in EHR discharge summary"),
    ],
    "Medium": [
        (False, "📅", "Schedule follow-up appointment within 7 days"),
        (False, "💊", "Review medications with patient; confirm understanding"),
        (False, "📞", "Provide patient with direct-contact number for concerns"),
        (False, "🏠", "Consider home health assessment referral"),
    ],
    "Low": [
        (False, "📅", "Schedule routine follow-up within 14 days"),
        (False, "📚", "Provide written discharge education materials"),
        (False, "✅", "Confirm patient understands warning signs for return"),
    ],
}


def _gauge_svg(score: int, risk: str) -> str:
    """SVG circular gauge showing score / 10."""
    colors = {
        "High":   ("#da3633", "#f85149"),
        "Medium": ("#9e6a03", "#d29922"),
        "Low":    ("#196c2e", "#3fb950"),
    }
    c1, c2 = colors.get(risk, ("#555", "#888"))
    r, cx, cy = 44, 55, 55
    circ = 2 * 3.14159 * r
    dash  = (score / 10) * circ
    offset = circ / 4
    return f"""
    <div class="gauge-wrap">
      <svg width="110" height="110" viewBox="0 0 110 110">
        <defs>
          <linearGradient id="g{risk}" x1="0%" y1="0%" x2="100%" y2="100%">
            <stop offset="0%" stop-color="{c1}"/>
            <stop offset="100%" stop-color="{c2}"/>
          </linearGradient>
        </defs>
        <circle cx="{cx}" cy="{cy}" r="{r}" fill="none" stroke="#21262d" stroke-width="10.5"/>
        <circle cx="{cx}" cy="{cy}" r="{r}" fill="none"
          stroke="url(#g{risk})" stroke-width="10.5"
          stroke-dasharray="{dash:.1f} {circ:.1f}"
          stroke-dashoffset="{offset:.1f}"
          stroke-linecap="round" style="transition:stroke-dasharray 0.6s ease"/>
        <text x="{cx}" y="51" text-anchor="middle"
          font-size="22" font-weight="800" fill="#e6edf3" font-family="Inter,sans-serif">{score}</text>
        <text x="{cx}" y="68" text-anchor="middle"
          font-size="10" fill="#8b949e" font-family="Inter,sans-serif">/ 10</text>
      </svg>
      <span class="gauge-label">Risk Score</span>
    </div>"""


def _big_badge(risk: str) -> str:
    """Large coloured risk badge with icon."""
    icons = {"High": "🔴", "Medium": "🟡", "Low": "🟢"}
    cls   = {"High": "bb-H", "Medium": "bb-M", "Low": "bb-L"}
    return (f'<span class="big-badge {cls.get(risk,"bb-L")}">'
            f'{icons.get(risk,"⚪")} {risk.upper()} RISK</span>')


def _score_bar(score: int, risk: str) -> str:
    """Horizontal score bar."""
    clr = {"High": "#f85149", "Medium": "#e3b341", "Low": "#3fb950"}
    return (f'<div class="score-bar-outer">'
            f'<div class="score-bar-inner" style="width:{score*10}%;'
            f'background:{clr.get(risk,"#888")}"></div></div>')


def _mini_bar(score: int, risk: str) -> str:
    """Compact bar for the dashboard table."""
    clr = {"High": "#f85149", "Medium": "#e3b341", "Low": "#3fb950"}
    return (f'<div class="mini-bar-wrap">'
            f'<div class="mini-bar-bg"><div class="mini-bar-fill" '
            f'style="width:{score*10}%;background:{clr.get(risk,"#888")}"></div></div>'
            f'<span style="font-size:0.75rem;color:#6e7681;width:18px">{score}</span>'
            f'</div>')


def _render_full_assessment(risk: str, score: int, factors: list,
                             breakdown: dict, patient_id: str = "",
                             patient_data: dict = None) -> None:
    """Render the complete assessment card used in both Assessment and Chat tabs."""
    # ── Header ──────────────────────────────────────────────────────────────
    col_b, col_g = st.columns([5, 1])
    with col_b:
        pid_label = f"**{patient_id}** · " if patient_id and patient_id != "CONV" else ""
        st.markdown(
            f"<div style='font-size:0.8rem;color:#8b949e;margin-bottom:8px'>"
            f"{pid_label}30-Day Readmission Risk Assessment · "
            f"{datetime.now().strftime('%d %b %Y, %H:%M')}</div>",
            unsafe_allow_html=True)
        st.markdown(_big_badge(risk), unsafe_allow_html=True)
        st.markdown(_score_bar(score, risk), unsafe_allow_html=True)
    with col_g:
        st.markdown(_gauge_svg(score, risk), unsafe_allow_html=True)

    # ── Patient data strip ───────────────────────────────────────────────────
    if patient_data and str(patient_data.get("patient_id","")) != "CONV":
        p = patient_data
        fields = [
            ("🎂", "Age",               p.get("age", "—")),
            ("🏥", "Condition",         p.get("primary_condition", "—")),
            ("🛏️", "LOS (days)",        p.get("length_of_stay_days", "—")),
            ("🔄", "Prior Adm. (6mo)",  p.get("prior_admissions_6mo", "—")),
            ("🏠", "Discharge To",      p.get("discharge_destination", "—")),
            ("📅", "Follow-up",         p.get("follow_up_scheduled", "—")),
            ("💊", "Medications",       p.get("medication_count", "—")),
            ("🩺", "Comorbidities",     p.get("comorbidity_count", "—")),
        ]
        cells = "".join(
            f"<div class='pcell'><div class='pcell-val'>{ic} {v}</div>"
            f"<div class='pcell-lbl'>{lbl}</div></div>"
            for ic, lbl, v in fields
        )
        st.markdown(f"<div class='acard-section-title'>Patient Summary</div>", unsafe_allow_html=True)
        st.markdown(f"<div class='pstrip'>{cells}</div>", unsafe_allow_html=True)

    # ── Two-column: factors + actions ────────────────────────────────────────
    col_f, col_a = st.columns(2)
    with col_f:
        st.markdown("<div class='acard-section-title'>🔍 Contributing Risk Factors</div>",
                    unsafe_allow_html=True)
        if factors:
            st.markdown(
                "".join(f"<span class='chip'>⚡ {f}</span>" for f in factors),
                unsafe_allow_html=True)
        else:
            st.markdown("<span class='chip chip-ok'>✅ No significant risk factors</span>",
                        unsafe_allow_html=True)

        # Score breakdown
        if breakdown:
            triggered = {k: v for k, v in breakdown.items() if v > 0}
            if triggered:
                bd_md = "| Rule | Points |\n|---|---|\n" + "\n".join(
                    f"| {k.replace('_',' ').title()} | +{v} |"
                    for k, v in triggered.items())
                with st.expander("📊 Score breakdown"):
                    st.markdown(bd_md)

    with col_a:
        st.markdown("<div class='acard-section-title'>📋 Recommended Actions</div>",
                    unsafe_allow_html=True)
        for is_esc, icon, text in _ACTIONS.get(risk, []):
            cls = "action action-esc" if is_esc else "action"
            st.markdown(f"<div class='{cls}'><span>{icon}</span>{text}</div>",
                        unsafe_allow_html=True)

    # ── Clinical notes input ─────────────────────────────────────────────────
    notes_key = f"notes_{patient_id or 'new'}_{score}"
    st.markdown("<div class='acard-section-title'>📝 Clinical Notes (optional)</div>",
                unsafe_allow_html=True)
    st.text_area("Add notes for handover:", key=notes_key, height=70,
                 placeholder="e.g. Patient lives alone, daughter contacted, follow-up booked for Monday…",
                 label_visibility="collapsed")

    # ── Disclaimers ──────────────────────────────────────────────────────────
    st.markdown(
        "<div class='disc'>⚠️ <strong>DISCLAIMER:</strong> This is a clinical "
        "decision-support tool only. Outputs are advisory and must be reviewed by "
        "a licensed clinician before any action is taken. This tool does not provide "
        "medical diagnosis or prescribe treatments.</div>",
        unsafe_allow_html=True)
    if risk == "High":
        st.markdown(
            "<div class='disc disc-esc'>🚨 <strong>ESCALATION REQUIRED:</strong> "
            "This patient has been flagged HIGH RISK for 30-day readmission. "
            "Immediate care coordinator review is required before discharge. "
            "Document in EHR accordingly.</div>",
            unsafe_allow_html=True)


def _render_checklist(patient_id: str, risk: str) -> None:
    """Interactive discharge checklist that persists in session state."""
    key = patient_id or "new"
    if key not in st.session_state.checklist:
        st.session_state.checklist[key] = {}

    st.markdown("<div class='acard-section-title'>✅ Discharge Checklist</div>",
                unsafe_allow_html=True)
    items = [a[2] for a in _ACTIONS.get(risk, [])]
    all_done = True
    for i, item in enumerate(items):
        done = st.session_state.checklist[key].get(i, False)
        if not done:
            all_done = False
        col_cb, col_txt = st.columns([0.5, 9.5])
        with col_cb:
            checked = st.checkbox("", value=done, key=f"cl_{key}_{i}")
            st.session_state.checklist[key][i] = checked
        with col_txt:
            cls = "cl-item cl-done" if checked else "cl-item"
            st.markdown(f"<div class='{cls}'>{item}</div>", unsafe_allow_html=True)

    if all_done and items:
        st.success("✅ All discharge tasks completed for this patient.")


def _build_report(pid: str, risk: str, score: int, factors: list,
                  patient_data: dict, notes: str = "") -> str:
    """Generate a plain-text printable discharge report."""
    now = datetime.now().strftime("%d %B %Y at %H:%M")
    p = patient_data or {}
    factor_lines = "\n".join(f"  • {f}" for f in factors) if factors else "  • None"
    actions_lines = "\n".join(f"  {i+1}. {a[2]}" for i, a in enumerate(_ACTIONS.get(risk, [])))
    return f"""
╔══════════════════════════════════════════════════════════════════╗
║       HOSPITAL READMISSION RISK ASSESSMENT REPORT               ║
║       System: Discharge Planning Hub (Decision-Support Only)     ║
╚══════════════════════════════════════════════════════════════════╝

Generated   : {now}
Patient ID  : {pid or "Ad-hoc Assessment"}
Risk Level  : {risk.upper()}  (Score: {score} / 10)

─── PATIENT SUMMARY ────────────────────────────────────────────────
Age                    : {p.get("age", "—")}
Primary Condition      : {p.get("primary_condition", "—")}
Length of Stay (days)  : {p.get("length_of_stay_days", "—")}
Prior Admissions (6mo) : {p.get("prior_admissions_6mo", "—")}
Discharge Destination  : {p.get("discharge_destination", "—")}
Follow-up Scheduled    : {p.get("follow_up_scheduled", "—")}
Medication Count       : {p.get("medication_count", "—")}
Comorbidity Count      : {p.get("comorbidity_count", "—")}

─── CONTRIBUTING RISK FACTORS ──────────────────────────────────────
{factor_lines}

─── RECOMMENDED ACTIONS ────────────────────────────────────────────
{actions_lines}

─── CLINICAL NOTES ─────────────────────────────────────────────────
{notes.strip() or "(No notes recorded)"}

─── DISCLAIMER ─────────────────────────────────────────────────────
This report is generated by an AI-assisted decision-support tool.
It does NOT constitute a clinical diagnosis and must NOT be used
as the sole basis for clinical decisions. All outputs must be
reviewed and confirmed by a licensed healthcare professional.
{"[!] HIGH RISK FLAG — Immediate care coordinator review required." if risk == "High" else ""}
════════════════════════════════════════════════════════════════════
""".strip()


# ══════════════════════════════════════════════════════════════════════════════
# Sidebar
# ══════════════════════════════════════════════════════════════════════════════
with st.sidebar:
    st.markdown("""
    <div class="sb-logo">
      <div class="sb-logo-title">🏥 Discharge Hub</div>
      <div class="sb-logo-sub">Readmission Risk Agent</div>
    </div>""", unsafe_allow_html=True)

    # Real-time stats
    st.markdown("<div class='sb-section'>Today's Summary</div>", unsafe_allow_html=True)
    try:
        df_s = _all_patients()
        n_high = (df_s["risk_level"] == "High").sum()
        n_med  = (df_s["risk_level"] == "Medium").sum()
        n_low  = (df_s["risk_level"] == "Low").sum()
        st.markdown(f"""
        <div class='sb-stat'><span class='sb-stat-l'>🔴 High Risk</span>
            <span class='sb-stat-v' style='color:#f85149'>{n_high}</span></div>
        <div class='sb-stat'><span class='sb-stat-l'>🟡 Medium Risk</span>
            <span class='sb-stat-v' style='color:#e3b341'>{n_med}</span></div>
        <div class='sb-stat'><span class='sb-stat-l'>🟢 Low Risk</span>
            <span class='sb-stat-v' style='color:#3fb950'>{n_low}</span></div>
        <div class='sb-stat'><span class='sb-stat-l'>👥 Total Patients</span>
            <span class='sb-stat-v'>{len(df_s)}</span></div>
        """, unsafe_allow_html=True)
    except Exception:
        st.warning("Dataset unavailable")

    st.markdown("<div class='sb-section'>Quick Actions</div>", unsafe_allow_html=True)
    if st.button("▶  Run Demo (P007 — Max Risk)"):
        st.session_state["demo_trigger"] = True
        st.rerun()
    if st.button("🗑  Clear All History"):
        st.session_state.chat_history  = []
        st.session_state.assess_result = None
        st.session_state.dashboard_sel = None
        st.session_state.checklist     = {}
        st.rerun()

    st.markdown("<div class='sb-section'>Scoring Rules</div>", unsafe_allow_html=True)
    st.markdown("""
    <div style='font-size:0.75rem;color:#8b949e;line-height:2.1'>
    +2 &nbsp; Age ≥ 75<br>
    +2 &nbsp; Prior admissions ≥ 2 (6mo)<br>
    +2 &nbsp; No follow-up scheduled<br>
    +1 &nbsp; LOS > 7 days<br>
    +1 &nbsp; Comorbidities ≥ 3<br>
    +1 &nbsp; Medications ≥ 8<br>
    +1 &nbsp; Discharged home (no services)<br>
    </div>
    <div style='font-size:0.72rem;color:#484f58;margin-top:10px;line-height:1.7'>
    🔴 High: 7–10 · 🟡 Medium: 4–6 · 🟢 Low: 0–3
    </div>""", unsafe_allow_html=True)

    st.markdown("<hr>", unsafe_allow_html=True)
    st.markdown(
        "<div style='font-size:0.68rem;color:#484f58;line-height:1.7'>"
        "⚠️ Decision-support only<br>"
        "Not a clinical diagnosis tool<br>"
        "LangChain + Ollama (llama3.2:3b)<br>"
        "Runs fully offline"
        "</div>", unsafe_allow_html=True)


# ══════════════════════════════════════════════════════════════════════════════
# Page header
# ══════════════════════════════════════════════════════════════════════════════
st.markdown(f"""
<div class="page-header">
  <div class="ph-badge">● OFFLINE · AI-ASSISTED · DECISION-SUPPORT ONLY</div>
  <div class="ph-title">🏥 Hospital Discharge Planning Hub</div>
  <div class="ph-sub">Identify high-risk patients before discharge · Plan targeted post-discharge interventions · Fully offline</div>
  <div class="ph-meta">
    Care Coordinator Tool &nbsp;·&nbsp;
    {datetime.now().strftime("%A, %d %B %Y")} &nbsp;·&nbsp;
    Powered by LangChain + Ollama (llama3.2:3b)
  </div>
</div>
""", unsafe_allow_html=True)

# ══════════════════════════════════════════════════════════════════════════════
# Tabs
# ══════════════════════════════════════════════════════════════════════════════
tab_dash, tab_assess, tab_chat, tab_data = st.tabs([
    "📊  Risk Dashboard",
    "🧑‍⚕️  Assess Patient",
    "💬  AI Assistant",
    "📋  Patient Records",
])


# ══════════════════════════════════════════════════════════════════════════════
# TAB 1 — Risk Dashboard
# ══════════════════════════════════════════════════════════════════════════════
with tab_dash:
    try:
        df_all = _all_patients()
    except Exception as e:
        st.error(f"❌ Could not load patient data: {e}")
        st.stop()

    # ── KPI row ──────────────────────────────────────────────────────────────
    k1, k2, k3, k4 = st.columns(4)
    n_high = (df_all["risk_level"] == "High").sum()
    n_med  = (df_all["risk_level"] == "Medium").sum()
    n_low  = (df_all["risk_level"] == "Low").sum()
    avg_sc = df_all["risk_score"].mean()

    k1.metric("🔴 High Risk Patients",   n_high, help="Require immediate intervention before discharge")
    k2.metric("🟡 Medium Risk Patients", n_med,  help="Require enhanced discharge planning")
    k3.metric("🟢 Low Risk Patients",    n_low,  help="Standard discharge process")
    k4.metric("📊 Avg Risk Score",       f"{avg_sc:.1f} / 10")

    st.markdown("")

    # ── Filter + sort ─────────────────────────────────────────────────────────
    fc1, fc2, fc3 = st.columns([2, 2, 3])
    with fc1:
        filter_risk = st.multiselect("Filter by risk level",
            ["High", "Medium", "Low"], default=[], key="dash_risk_filter")
    with fc2:
        filter_cond = st.multiselect("Filter by condition",
            sorted(df_all["primary_condition"].unique().tolist()), default=[], key="dash_cond_filter")
    with fc3:
        sort_by = st.selectbox("Sort by",
            ["Risk Score (High → Low)", "Risk Score (Low → High)", "Patient ID", "Age (Oldest first)"],
            key="dash_sort")

    df_view = df_all.copy()
    if filter_risk:
        df_view = df_view[df_view["risk_level"].isin(filter_risk)]
    if filter_cond:
        df_view = df_view[df_view["primary_condition"].isin(filter_cond)]

    sort_map = {
        "Risk Score (High → Low)":  ("risk_score", False),
        "Risk Score (Low → High)":  ("risk_score", True),
        "Patient ID":               ("patient_id", True),
        "Age (Oldest first)":       ("age", False),
    }
    scol, sasc = sort_map.get(sort_by, ("risk_score", False))
    df_view = df_view.sort_values(scol, ascending=sasc)

    st.markdown(f"<div style='font-size:0.78rem;color:#8b949e;margin-bottom:10px'>"
                f"Showing {len(df_view)} patients</div>", unsafe_allow_html=True)

    # ── Table header ──────────────────────────────────────────────────────────
    st.markdown("""
    <div class='risk-row risk-row-header'>
        <span>Patient ID</span><span>Age</span><span>Condition</span>
        <span>Risk Level</span><span>LOS (d)</span>
        <span>Follow-up</span><span>Discharge</span><span>Score</span>
    </div>""", unsafe_allow_html=True)

    badge_cls = {"High": "rb rb-H", "Medium": "rb rb-M", "Low": "rb rb-L"}
    fu_icon   = {"Yes": "✅ Yes", "No": "❌ No"}

    for _, row in df_view.iterrows():
        pid   = row["patient_id"]
        risk  = row["risk_level"]
        score = row["risk_score"]
        rb    = f"<span class='{badge_cls[risk]}'>{_RISK_ICON[risk]} {risk}</span>"
        bar   = _mini_bar(score, risk)
        fu    = fu_icon.get(str(row["follow_up_scheduled"]), row["follow_up_scheduled"])

        col_row, col_btn = st.columns([11, 1])
        with col_row:
            st.markdown(f"""
            <div class='risk-row'>
                <span style='color:#e6edf3;font-weight:600'>{pid}</span>
                <span>{row["age"]}</span>
                <span>{row["primary_condition"]}</span>
                {rb}
                <span>{row["length_of_stay_days"]}</span>
                <span style='font-size:0.8rem'>{fu}</span>
                <span style='color:#8b949e;font-size:0.8rem'>{row["discharge_destination"]}</span>
                {bar}
            </div>""", unsafe_allow_html=True)
        with col_btn:
            if st.button("View", key=f"dash_view_{pid}"):
                st.session_state.dashboard_sel = (
                    None if st.session_state.dashboard_sel == pid else pid
                )
                st.rerun()

        # Inline expanded assessment
        if st.session_state.dashboard_sel == pid:
            with st.container():
                st.markdown(
                    f"<div style='background:#0d1117;border:1px solid #21262d;border-radius:12px;"
                    f"padding:20px 22px;margin:4px 0 12px'>",
                    unsafe_allow_html=True)
                _render_full_assessment(
                    risk  = row["risk_level"],
                    score = row["risk_score"],
                    factors = row["factors"],
                    breakdown = row["breakdown"],
                    patient_id = pid,
                    patient_data = row.to_dict(),
                )
                _render_checklist(pid, row["risk_level"])

                # Report generate
                notes_key = f"notes_{pid}_{score}"
                notes_val = st.session_state.get(notes_key, "")
                report = _build_report(
                    pid, row["risk_level"], row["risk_score"],
                    row["factors"], row.to_dict(), notes=notes_val
                )
                st.download_button(
                    "⬇️  Download Discharge Report (.txt)",
                    data=report,
                    file_name=f"discharge_report_{pid}_{datetime.now().strftime('%Y%m%d_%H%M')}.txt",
                    mime="text/plain",
                    key=f"dl_{pid}",
                )
                st.markdown("</div>", unsafe_allow_html=True)


# ══════════════════════════════════════════════════════════════════════════════
# TAB 2 — Assess Patient
# ══════════════════════════════════════════════════════════════════════════════
with tab_assess:

    # Demo trigger from sidebar
    demo_triggered = st.session_state.pop("demo_trigger", False)

    mode_col, _ = st.columns([3, 4])
    with mode_col:
        entry_mode = st.radio(
            "Patient entry method",
            ["📂 Select existing patient (P001–P040)", "➕ Enter new patient details"],
            horizontal=True,
            key="assess_mode",
        )

    st.markdown("")

    # ─────────────────────────────────────────────────────────────────────────
    if "existing" in entry_mode or demo_triggered:
        # Existing patient
        ec1, ec2 = st.columns([2, 1])
        with ec1:
            try:
                ids = _get_ids()
            except Exception:
                ids = []
            default_idx = ids.index("P007") if "P007" in ids else 0
            sel_pid = st.selectbox(
                "Select Patient ID",
                ids,
                index=default_idx if demo_triggered else 0,
                key="assess_pid_sel",
            )
        with ec2:
            st.markdown("<br>", unsafe_allow_html=True)
            run_btn = st.button("🔍 Assess Patient", key="assess_run", use_container_width=True)

        if run_btn or demo_triggered:
            with st.spinner(f"Scoring {sel_pid}…"):
                result = run_agent_direct(sel_pid)
                if result["error"]:
                    st.error(f"❌ Patient **{sel_pid}** not found.")
                else:
                    st.session_state.assess_result = result
            st.rerun()

        if st.session_state.assess_result:
            r = st.session_state.assess_result
            st.markdown("---")
            _render_full_assessment(
                risk=r["risk_assessment"]["risk_level"],
                score=r["risk_assessment"]["score"],
                factors=r["risk_assessment"]["contributing_factors"],
                breakdown=r["risk_assessment"]["score_breakdown"],
                patient_id=r["patient_id"],
                patient_data=r["patient_data"],
            )
            _render_checklist(r["patient_id"], r["risk_assessment"]["risk_level"])

            notes_key = f"notes_{r['patient_id']}_{r['risk_assessment']['score']}"
            notes_val = st.session_state.get(notes_key, "")
            report    = _build_report(
                r["patient_id"], r["risk_assessment"]["risk_level"],
                r["risk_assessment"]["score"],
                r["risk_assessment"]["contributing_factors"],
                r["patient_data"], notes=notes_val,
            )
            st.download_button(
                "⬇️  Download Discharge Report (.txt)",
                data=report,
                file_name=f"discharge_report_{r['patient_id']}_{datetime.now().strftime('%Y%m%d_%H%M')}.txt",
                mime="text/plain",
                key="dl_assess",
            )
        else:
            st.markdown("")
            st.markdown(
                "<div class='empty'><div class='empty-icon'>🧑‍⚕️</div>"
                "<div class='empty-title'>No assessment yet</div>"
                "<div class='empty-sub'>Select a patient ID and click Assess Patient.</div></div>",
                unsafe_allow_html=True)

    else:
        # ── New patient form ─────────────────────────────────────────────────
        st.markdown(
            "<div class='info-banner'>Fill in all available fields. "
            "The risk score updates automatically when you click <strong>Calculate Risk</strong>.</div>",
            unsafe_allow_html=True)

        with st.form("new_patient_form"):
            fc1, fc2, fc3 = st.columns(3)
            with fc1:
                f_age  = st.number_input("Patient Age", 18, 110, 65, key="f_age")
                f_los  = st.number_input("Length of Stay (days)", 0, 90, 5, key="f_los")
                f_meds = st.number_input("Medication Count", 0, 30, 5, key="f_meds")
            with fc2:
                f_cond   = st.selectbox("Primary Condition",
                    ["Heart Failure","COPD","Diabetes","Pneumonia","Hip Fracture","Other"], key="f_cond")
                f_prior  = st.number_input("Prior Admissions (last 6 months)", 0, 10, 0, key="f_prior")
                f_comor  = st.number_input("Comorbidity Count", 0, 10, 0, key="f_comor")
            with fc3:
                f_disc  = st.selectbox("Discharge Destination",
                    ["Home", "Home+Services", "SNF", "Rehab"], key="f_disc")
                f_fu    = st.selectbox("Follow-up Appointment Scheduled?",
                    ["Yes", "No"], key="f_fu")
                f_name  = st.text_input("Patient Reference (optional)",
                    placeholder="e.g. New Admit, Bed 14…", key="f_name")

            submitted = st.form_submit_button("⚡ Calculate Risk Score", use_container_width=True)

        if submitted:
            patient_dict = {
                "patient_id":             f_name or "NEW",
                "age":                    f_age,
                "primary_condition":      f_cond,
                "length_of_stay_days":    f_los,
                "prior_admissions_6mo":   f_prior,
                "discharge_destination":  f_disc,
                "follow_up_scheduled":    f_fu,
                "medication_count":       f_meds,
                "comorbidity_count":      f_comor,
            }
            a = compute_risk_score(patient_dict)
            st.session_state.assess_result = {
                "patient_id":    f_name or "NEW",
                "patient_data":  patient_dict,
                "risk_assessment": a,
                "error": None,
            }
            st.rerun()

        if st.session_state.assess_result and "NEW" in str(st.session_state.assess_result.get("patient_id", "")):
            r = st.session_state.assess_result
            st.markdown("---")
            _render_full_assessment(
                risk=r["risk_assessment"]["risk_level"],
                score=r["risk_assessment"]["score"],
                factors=r["risk_assessment"]["contributing_factors"],
                breakdown=r["risk_assessment"]["score_breakdown"],
                patient_id=r["patient_id"],
                patient_data=r["patient_data"],
            )
            _render_checklist(r["patient_id"], r["risk_assessment"]["risk_level"])

            notes_key = f"notes_{r['patient_id']}_{r['risk_assessment']['score']}"
            notes_val = st.session_state.get(notes_key, "")
            report    = _build_report(
                r["patient_id"], r["risk_assessment"]["risk_level"],
                r["risk_assessment"]["score"],
                r["risk_assessment"]["contributing_factors"],
                r["patient_data"], notes=notes_val,
            )
            st.download_button(
                "⬇️  Download Discharge Report (.txt)", data=report,
                file_name=f"discharge_report_{datetime.now().strftime('%Y%m%d_%H%M')}.txt",
                mime="text/plain", key="dl_new",
            )


# ══════════════════════════════════════════════════════════════════════════════
# TAB 3 — AI Assistant
# ══════════════════════════════════════════════════════════════════════════════
with tab_chat:

    # Suggested prompts
    st.markdown(
        "<div style='font-size:0.82rem;color:#8b949e;margin-bottom:10px'>"
        "💡 Quick prompts — click to run instantly:</div>",
        unsafe_allow_html=True)

    prompts = [
        ("Assess P007", "🔴 High Risk patient"),
        ("65 year old, COPD, 3 prior admissions, no follow-up", "Free text"),
        ("Assess P039", "🟢 Low Risk patient"),
        ("85yo Heart Failure, 18 day stay, 5 prior admissions, no follow-up, home discharge, 14 meds, 5 comorbidities", "Worst case"),
    ]
    pc = st.columns(len(prompts))
    clicked_prompt = None
    for i, (col, (ptxt, plabel)) in enumerate(zip(pc, prompts)):
        with col:
            if st.button(f"**{plabel}**\n\n_{ptxt[:35]}…_" if len(ptxt) > 35 else f"**{plabel}**\n\n_{ptxt}_",
                         key=f"prompt_{i}", use_container_width=True):
                clicked_prompt = ptxt

    st.markdown("")

    # ── Chat history ──────────────────────────────────────────────────────────
    for msg in st.session_state.chat_history[-12:]:
        if msg["role"] == "user":
            st.markdown(
                f"<div class='bubble-u'><div class='bubble-u-inner'>🧑‍💼 {msg['content']}</div></div>",
                unsafe_allow_html=True)
        else:
            st.markdown(
                f"<div class='bubble-a'><div class='bubble-a-av'>🤖</div>"
                f"<div class='bubble-a-inner'>{msg['content']}</div></div>",
                unsafe_allow_html=True)
            if msg.get("assessment"):
                _render_full_assessment(
                    risk=msg["assessment"]["risk_level"],
                    score=msg["assessment"]["score"],
                    factors=msg["assessment"]["contributing_factors"],
                    breakdown=msg["assessment"]["score_breakdown"],
                    patient_id=msg.get("pid",""),
                    patient_data=msg.get("pdata"),
                )

    if not st.session_state.chat_history:
        st.markdown(
            "<div class='empty'><div class='empty-icon'>💬</div>"
            "<div class='empty-title'>Ask the AI Assistant</div>"
            "<div class='empty-sub'>Type a patient ID or describe a patient — "
            "or click a quick prompt above.</div></div>",
            unsafe_allow_html=True)

    # ── Input ─────────────────────────────────────────────────────────────────
    with st.form("chat_form", clear_on_submit=True):
        c_inp, c_btn = st.columns([7, 1])
        with c_inp:
            user_msg = st.text_area("Message", height=72, label_visibility="collapsed",
                placeholder="e.g.  Assess patient P012  /  78yo Heart Failure, 12 day stay, no follow-up…")
        with c_btn:
            st.markdown("<div style='height:14px'></div>", unsafe_allow_html=True)
            send_btn = st.form_submit_button("Send ➤", use_container_width=True)

    query = clicked_prompt or (user_msg.strip() if send_btn and user_msg else None)

    if query:
        st.session_state.chat_history.append({"role": "user", "content": query})
        with st.spinner("🧠 Assessing…"):
            pid_match = re.search(r'\bP\d{3}\b', query.upper())
            try:
                if pid_match:
                    pid    = pid_match.group(0)
                    result = run_agent_direct(pid)
                    if result["error"]:
                        st.session_state.chat_history.append({
                            "role": "agent",
                            "content": f"❌ Patient **{pid}** not found. Valid IDs: P001–P040.",
                        })
                    else:
                        a   = result["risk_assessment"]
                        msg = (f"Assessment complete for **{pid}**. "
                               f"Risk: **{a['risk_level']}** · Score: **{a['score']}/10**")
                        st.session_state.chat_history.append({
                            "role": "agent", "content": msg,
                            "assessment": a, "pid": pid, "pdata": result["patient_data"],
                        })
                else:
                    result = run_agent_conversational(query)
                    a   = result["risk_assessment"]
                    p   = result["parsed_fields"]
                    msg = (f"Based on the description: Risk **{a['risk_level']}** "
                           f"(Score {a['score']}/10). "
                           f"Extracted — Age: {p['age']}, Condition: {p['primary_condition']}, "
                           f"Prior admissions: {p['prior_admissions_6mo']}, "
                           f"Follow-up: {p['follow_up_scheduled']}.")
                    st.session_state.chat_history.append({
                        "role": "agent", "content": msg,
                        "assessment": a, "pid": "", "pdata": p,
                    })
            except Exception as e:
                st.session_state.chat_history.append({
                    "role": "agent", "content": f"❌ Error: {e}",
                })
        st.rerun()


# ══════════════════════════════════════════════════════════════════════════════
# TAB 4 — Patient Records
# ══════════════════════════════════════════════════════════════════════════════
with tab_data:
    try:
        df_rec = _all_patients()
    except Exception as e:
        st.error(f"❌ Could not load: {e}")
        st.stop()

    # Filters
    rf1, rf2, rf3, rf4 = st.columns(4)
    with rf1:
        f_rl = st.multiselect("Risk Level",   ["High","Medium","Low"],     key="r_rl")
    with rf2:
        f_cn = st.multiselect("Condition",    sorted(df_rec["primary_condition"].unique()), key="r_cn")
    with rf3:
        f_fu = st.multiselect("Follow-up",    ["Yes","No"],                key="r_fu")
    with rf4:
        f_ag = st.slider("Age range", int(df_rec["age"].min()),
                         int(df_rec["age"].max()),
                         (int(df_rec["age"].min()), int(df_rec["age"].max())), key="r_ag")

    df_r = df_rec.copy()
    if f_rl: df_r = df_r[df_r["risk_level"].isin(f_rl)]
    if f_cn: df_r = df_r[df_r["primary_condition"].isin(f_cn)]
    if f_fu: df_r = df_r[df_r["follow_up_scheduled"].isin(f_fu)]
    df_r = df_r[df_r["age"].between(f_ag[0], f_ag[1])]
    df_r = df_r.sort_values("risk_score", ascending=False)

    # Summary strip
    sm1, sm2, sm3, sm4, sm5 = st.columns(5)
    sm1.metric("Showing",       len(df_r))
    sm2.metric("High Risk",     (df_r["risk_level"]=="High").sum() if len(df_r) else 0)
    sm3.metric("Avg Score",     f"{df_r['risk_score'].mean():.1f}" if len(df_r) else "—")
    sm4.metric("Avg Age",       f"{df_r['age'].mean():.0f}" if len(df_r) else "—")
    sm5.metric("Avg LOS",       f"{df_r['length_of_stay_days'].mean():.1f}d" if len(df_r) else "—")

    st.markdown("")

    # Display columns
    display_cols = ["patient_id","age","primary_condition","risk_level","risk_score",
                    "length_of_stay_days","prior_admissions_6mo",
                    "discharge_destination","follow_up_scheduled",
                    "medication_count","comorbidity_count","readmitted_30d"]
    df_display = df_r[[c for c in display_cols if c in df_r.columns]]

    st.dataframe(
        df_display,
        use_container_width=True,
        hide_index=True,
        height=500,
        column_config={
            "patient_id":              st.column_config.TextColumn("ID",          width=70),
            "age":                     st.column_config.NumberColumn("Age",        width=60),
            "primary_condition":       st.column_config.TextColumn("Condition",    width=130),
            "risk_level":              st.column_config.TextColumn("Risk",         width=85),
            "risk_score":              st.column_config.ProgressColumn("Score", min_value=0, max_value=10, width=100),
            "length_of_stay_days":     st.column_config.NumberColumn("LOS (d)",    width=75),
            "prior_admissions_6mo":    st.column_config.NumberColumn("Prior Adm.", width=85),
            "discharge_destination":   st.column_config.TextColumn("Discharge",    width=110),
            "follow_up_scheduled":     st.column_config.TextColumn("Follow-up",    width=85),
            "medication_count":        st.column_config.NumberColumn("Meds",       width=65),
            "comorbidity_count":       st.column_config.NumberColumn("Comorbid.",  width=80),
            "readmitted_30d":          st.column_config.TextColumn("Readmitted",   width=90),
        },
    )

    # Export
    csv_export = df_display.to_csv(index=False)
    st.download_button(
        "⬇️  Export filtered data as CSV",
        data=csv_export,
        file_name=f"patient_records_{datetime.now().strftime('%Y%m%d_%H%M')}.csv",
        mime="text/csv",
    )
    st.caption(
        f"📌 {len(df_r)} records shown · Sorted by risk score · "
        "Synthetic/anonymised data only · For demonstration purposes"
    )
