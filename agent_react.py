# Alternate ReAct agent entrypoint (not used by the Streamlit UI).
# Requires Ollama to be running. Import from here if needed for CLI use.
from __future__ import annotations

import os
from typing import Union

from langchain.agents import AgentExecutor, create_react_agent
from langchain.prompts import PromptTemplate
from langchain_ollama import ChatOllama

from agent import (
    MODEL_NAME,
    _load_system_prompt,
    patient_lookup_tool,
)
from tools.risk_scorer import risk_scorer_tool
from utils.data_loader import load_patient_by_id
from tools.risk_scorer import compute_risk_score


def _build_agent() -> AgentExecutor:
    system_text = _load_system_prompt()

    react_template = (
        system_text
        + """

You have access to the following tools:

{tools}

Use the following format EXACTLY - do not deviate:

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
- Do NOT guess or invent a risk score - always use the tool result.

Begin!

Question: {input}
Thought:{agent_scratchpad}"""
    )

    prompt = PromptTemplate.from_template(react_template)

    llm = ChatOllama(
        model=MODEL_NAME,
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


_agent_executor: Union[AgentExecutor, None] = None


def _get_executor() -> AgentExecutor:
    global _agent_executor
    if _agent_executor is None:
        _agent_executor = _build_agent()
    return _agent_executor


def run_agent(query: str) -> str:
    if not query or not query.strip():
        return (
            "No input provided. Please enter a patient ID or patient details "
            "to generate a readmission risk assessment."
        )

    try:
        executor = _get_executor()
        result = executor.invoke({"input": query})
        output = result.get("output", "")

        disclaimer = (
            "\n\nDISCLAIMER: This is a decision-support tool. "
            "Always consult a licensed clinician before acting on this output."
        )
        if "disclaimer" not in output.lower() and "advisory" not in output.lower():
            output += disclaimer

        return output

    except Exception as exc:
        err = str(exc)
        if "connection" in err.lower() or "refused" in err.lower():
            return (
                "Cannot connect to Ollama.\n\n"
                "Please ensure Ollama is running:\n"
                "  1. Open a terminal\n"
                "  2. Run: `ollama serve`\n"
                "  3. In another terminal, verify: `ollama list`\n"
                "  4. Confirm llama3:8b is listed, then retry."
            )
        return f"Agent error: {err}"


def run_agent_direct(patient_id: str) -> dict:
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
