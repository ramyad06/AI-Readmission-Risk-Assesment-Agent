# Demo Script — Hospital Readmission Prediction Agent

> **Duration:** ~5 minutes | **Target Audience:** Evaluators, supervisors, internship panel

---

## Pre-Demo Checklist

- [ ] Ollama installed and running (`ollama serve`)
- [ ] `llama3:8b` model pulled (`ollama list` shows it)
- [ ] Virtual environment activated (`source venv/bin/activate`)
- [ ] Streamlit app started (`streamlit run app.py`)
- [ ] Browser open at `http://localhost:8501`

---

## Step 1: Start Ollama and Streamlit

```bash
# Terminal 1 — start the Ollama model server
ollama serve

# Terminal 2 — activate venv and launch the UI
cd readmission_agent
source venv/bin/activate
streamlit run app.py
```

**Narrator (30 sec):**
> "This agent runs completely offline — no cloud, no internet, no data leaving
> the hospital network. Ollama hosts the llama3:8b language model locally.
> Streamlit provides the interactive UI. Everything you're about to see runs
> on a standard laptop."

---

## Step 2: Demo Scenario A — High Risk Patient via CSV Mode

**Actions:**
1. In the sidebar, select **"Mode B — CSV Patient Lookup"**
2. From the dropdown, select **P007**
3. Click **"Assess Patient"**

*(Alternatively click the **"▶ Run Demo Patient (P007)"** button at the top of any page.)*

**Expected Output:**
- 🔴 **HIGH RISK** badge with pulsing red glow
- Score: **10 / 10**
- Contributing Factors: all 7 rules triggered
- Actions: escalation language prominently displayed

**Narrator (30 sec):**
> "Patient P007 is 85 years old with Heart Failure — an 18-day stay, 5 prior
> admissions in the last 6 months, 14 medications, 5 comorbidities, and no
> follow-up scheduled. Every single risk factor fires — score is 10 out of 10.
> Notice the system immediately triggers escalation language: it recommends
> immediate care coordinator review. The disclaimer is always present."

---

## Step 3: Demo Scenario B — Conversational Input

**Actions:**
1. In the sidebar, select **"Mode A — Conversational Input"**
2. Type or paste the following into the text area:

```
Assess the readmission risk for a 72-year-old patient with Heart Failure.
They had a 13-day hospital stay, 3 prior admissions in the last 6 months,
no follow-up scheduled, 11 medications, 4 comorbidities, discharged home.
```

3. Click **"Submit Query"**

**Expected Output:**
- Agent calls `patient_lookup_tool` (or processes inline data)
- Agent calls `risk_scorer_tool` — score computed deterministically
- 🔴 **HIGH RISK** assessment with multi-step reasoning log

**Narrator (30 sec):**
> "In conversational mode, a care coordinator can describe a patient without
> selecting from a dropdown. Watch the agent's reasoning chain in the terminal —
> it first calls the lookup tool, then the scorer, then synthesises the
> assessment. This multi-step tool use is the core of the LangChain agent
> architecture — the LLM never guesses the score, it always calls the tool."

---

## Step 4: Disclaimer and Escalation Behaviour

**Actions:**
1. Point to the amber disclaimer box at the bottom of any result
2. Show the High Risk result for P007 — highlight the escalation block

**Narrator (30 sec):**
> "Every single response — without exception — includes a safety disclaimer
> reminding users that this is an advisory tool only. For High Risk patients
> there's a second, more prominent escalation warning. The agent is
> domain-constrained: it will never suggest specific medications or
> provide a clinical diagnosis. If you ask it to, it politely declines
> and redirects to a licensed clinician."

---

## Step 5: Edge Case — Young Patient Flagged Low Risk

**Actions:**
1. Remain in **Mode B — CSV Patient Lookup**
2. Select **P039** (age 24, COPD, short stay, follow-up scheduled)
3. Click **"Assess Patient"**

**Expected Output:**
- 🟢 **LOW RISK** badge
- Score: **2 / 10** (only 'No follow-up' if not scheduled — P039 has follow-up)
- Factors: minimal or none

**Narrator (30 sec):**
> "This is a 24-year-old COPD patient — short stay, no prior admissions,
> follow-up booked. Score is very low. This shows the system doesn't blindly
> flag everyone; it's calibrated. Compare this to P036 — a 90-year-old
> with Heart Failure — who scores the maximum 10. The scoring is fully
> deterministic, auditable, and explainable: no black-box ML involved."

---

## Step 6: Validation Run (optional — for technical evaluators)

**Actions:**
1. Open a terminal in the project directory
2. Run the batch validation:

```bash
python validate_agent.py
```

**Expected Output:**
- Per-patient table comparing predicted vs actual readmission
- Confusion matrix with precision, recall, F1
- Risk level distribution (roughly 30% High, 40% Medium, 30% Low)

**Narrator (30 sec):**
> "Finally, we can validate the entire dataset in seconds — no Ollama needed
> since this uses the deterministic scoring path directly. The output includes
> a confusion matrix, precision and recall figures. This is important for the
> healthcare setting: the tool needs to be auditable, not a magic black box."

---

## Talking Points Summary

| Step | Key Message |
|------|-------------|
| 1 | Fully offline — no cloud dependency |
| 2 | High Risk triggers escalation + pulsing badge |
| 3 | Multi-step tool calling, not LLM hallucination |
| 4 | Disclaimer present on every response |
| 5 | Calibrated — young healthy patients score Low |
| 6 | Batch-auditable, deterministic, no ML training |

---

## Anticipated Questions & Answers

**Q: Why not use a trained ML model?**
> A: Rule-based scoring is fully explainable and auditable — critical in
> regulated healthcare environments where clinicians must understand exactly
> why a risk flag was raised. ML models require labelled training data today
> and introduce drift over time.

**Q: What if Ollama is slow?**
> A: The deterministic scoring path (Mode B) works independently of the LLM.
> The UI renders the risk badge and score in under a second. The LLM narrative
> adds context, but is not required for the core assessment.

**Q: Can this be used in production?**
> A: Not as-is — this is a proof-of-concept / internship project. Production
> would require clinical validation, regulatory compliance (HIPAA, MDR),
> professional sign-off, and integration with real EHR systems.
