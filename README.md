# Hospital Discharge Planning Hub

Hospital Discharge Planning Hub is an offline-first decision-support prototype for estimating 30-day readmission risk from synthetic patient records.

## Overview

- Deterministic risk scoring engine with explainable rule breakdown (0 to 10)
- Local LLM-assisted reasoning for summaries and non-clinical preventive actions
- Streamlit interface with Dashboard, Assess Patient, AI Assistant, and Records views
- Graceful fallback to deterministic output when LLM/Ollama is unavailable

## How it works

1. Patient data is loaded from `data/patients.csv` via `utils/data_loader.py`.
2. `tools/risk_scorer.py` computes a transparent risk score and risk level.
3. `agent.py` orchestrates scoring plus optional local LLM reasoning.
4. `app.py` renders the Streamlit UI for exploration and interactive assessment.

## Project structure

- `app.py`: Streamlit application entry point
- `agent.py`: Assessment orchestration, role-aware reasoning, and fallback logic
- `tools/risk_scorer.py`: Rule-based scoring tool
- `utils/data_loader.py`: CSV data loading helpers
- `validate_agent.py`: Batch validation against dataset labels
- `tests/`: Unit tests for risk scoring and assessment output contract
- `prompts/system_prompt.txt`: System prompt used by the agent
- `data/patients.csv`: Synthetic dataset for demo/testing

## Quick start

1. Create and activate a virtual environment.
2. Install dependencies:

```bash
pip install -r requirements.txt
```

3.Start Ollama and pull models:

```bash
ollama serve
ollama pull llama3.2:1b
ollama pull llama3.2:3b
```

4. Run the app:

```bash
streamlit run app.py
```

## Safety note

This project is a decision-support prototype and not a clinical diagnosis system. Always involve licensed clinicians before acting on risk outputs.