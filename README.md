# Hospital Patient Readmission Prediction Agent

A fully offline, AI-powered clinical decision-support tool for assessing
30-day hospital readmission risk. Built as an internship project using
LangChain, Ollama (llama3:8b), Python 3, and Streamlit.

**This tool is for decision-support only. It is not a clinical diagnosis
system and does not provide medical advice.**

---

## Project Overview

Healthcare readmission within 30 days of discharge is a key quality indicator
and a major cost driver for hospitals. This agent helps care coordinators and
discharge planners identify high-risk patients before discharge, so that
appropriate interventions — such as scheduling follow-up appointments or
arranging home health services — can be put in place proactively.

The agent uses a transparent, rule-based scoring heuristic (0–10 points across
7 auditable clinical factors) combined with a locally-hosted large language model
(llama3:8b via Ollama) to generate natural-language risk assessments with
recommended actions. The entire stack runs offline — no patient data ever leaves
the hospital network.

---

## Architecture

```
┌─────────────────────────────────────────────────────────────┐
│                      Streamlit UI (app.py)                  │
│  ┌─────────────────────────┐  ┌──────────────────────────┐  │
│  │  Mode A: Conversational │  │  Mode B: CSV Patient     │  │
│  │  Input                  │  │  Lookup (dropdown)       │  │
│  └────────────┬────────────┘  └────────────┬─────────────┘  │
└───────────────┼─────────────────────────────┼───────────────┘
                │                             │
                ▼                             ▼
┌─────────────────────────────────────────────────────────────┐
│                  LangChain Agent (agent.py)                  │
│  AgentExecutor — ReAct reasoning loop                       │
│                                                             │
│  ┌───────────────────────┐  ┌───────────────────────────┐  │
│  │  patient_lookup_tool  │  │    risk_scorer_tool        │  │
│  │  (data_loader.py)     │  │    (risk_scorer.py)        │  │
│  └───────────┬───────────┘  └────────────┬──────────────┘  │
└──────────────┼──────────────────────────┼──────────────────┘
               │                          │
               ▼                          ▼
┌──────────────────────┐    ┌─────────────────────────────────┐
│  data/patients.csv   │    │  Rule Engine (deterministic)     │
│  40 synthetic records│    │  7 rules, 0–10 score, 3 levels  │
└──────────────────────┘    └─────────────────────────────────┘
               │
               ▼
┌─────────────────────────────────────────────────────────────┐
│              Ollama — llama3:8b (local inference)            │
│  No internet required. No cloud API. Fully air-gapped.      │
└─────────────────────────────────────────────────────────────┘
```

---

## Setup Instructions

### Prerequisites

- macOS / Linux
- Python 3.10+
- [Ollama](https://ollama.com/download) installed

### Step-by-Step

**1. Clone / extract the project**
```bash
cd readmission_agent
```

**2. Run the one-time setup script**
```bash
bash setup.sh
```
This creates the virtual environment, installs all packages, pulls `llama3:8b`,
and verifies the model is responding.

**3. Activate the virtual environment**
```bash
source venv/bin/activate
```

**4. Start Ollama (if not already running)**
```bash
ollama serve
```

**5. Launch the Streamlit UI**
```bash
streamlit run app.py
```

Open `http://localhost:8501` in your browser.

---

## Usage Guide

### Mode A — Conversational Input
Type a natural language query in the text area:
```
Assess the readmission risk for patient P007.
```
or paste patient demographics directly:
```
72-year-old, Heart Failure, 13-day stay, 3 prior admissions,
no follow-up, 11 medications, 4 comorbidities, discharged home.
```

### Mode B — CSV Patient Lookup
Select a patient ID from the dropdown (P001–P040) and click **Assess Patient**.
The system renders a risk badge, contributing factors, recommended actions,
and a score breakdown table.

### Run Demo Patient
Click **"▶ Run Demo Patient (P007 — High Risk)"** to showcase the agent
with a maximum-risk example.

### Validation
Run the batch validation (no Ollama needed):
```bash
python validate_agent.py
```

---

## Tech Stack

| Component          | Technology                        | Version   |
|--------------------|-----------------------------------|-----------|
| Language Model     | Ollama — llama3:8b (local)        | 3.x       |
| Agent Framework    | LangChain + LangChain-Ollama      | 0.3.x     |
| UI Framework       | Streamlit                         | 1.45.x    |
| Data Handling      | pandas                            | 2.2.x     |
| Scoring Engine     | Pure Python (rule-based)          | N/A       |
| Runtime            | Python                            | 3.10+     |
| Model Server       | Ollama                            | Latest    |
| Deployment Target  | Local / On-premise (offline)      | —         |

---

## Known Limitations & Out-of-Scope Items

| Limitation | Notes |
|------------|-------|
| Synthetic data only | No real patient records. For production, EHR integration required. |
| 40-patient dataset | Demonstrates the concept; not statistically representative. |
| Rule-based scoring | Heuristic, not clinically validated. Requires clinical sign-off. |
| No authentication | UI has no login. Production would require RBAC/SSO. |
| Response time varies | LLM narrative depends on hardware. Score card is always instant. |
| English only | No multilingual support. |
| No audit logging | Production systems require immutable audit trails. |
| Static rules | Risk factors don't adapt to patient population trends. |

---

## Evaluation Checklist

| Criterion | Status |
|-----------|--------|
| Agent calls tool (no hallucinated score) | ✅ |
| Multi-step reasoning visible in agent logs | ✅ |
| Output includes all 4 required sections | ✅ |
| High Risk triggers escalation language | ✅ |
| Disclaimer present in every response | ✅ |
| Runs fully offline | ✅ |
| Response under 5 seconds (score card) | ✅ |
| CSV and conversational modes both work | ✅ |

---

## Safety Disclaimer

> **⚠️ IMPORTANT:** This tool is a proof-of-concept decision-support system
> developed as part of a healthcare IT internship. It is **not** approved for
> clinical use, does **not** constitute medical advice, and **must not** be used
> to make clinical decisions without review by a licensed healthcare professional.
>
> Patient data in this project is entirely synthetic and anonymised.
> No real patient information has been used or stored.
>
> For production deployment in a clinical environment, this tool would require:
> - Clinical validation and regulatory approval (e.g., FDA 510(k), CE mark)
> - HIPAA / GDPR compliance review
> - Integration with certified EHR systems
> - Clinical governance sign-off

---

*Built by: Healthcare IT Internship Project | Stack: LangChain + Ollama + Streamlit*
