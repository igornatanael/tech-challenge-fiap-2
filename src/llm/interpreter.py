from __future__ import annotations

from src.llm.client import chat
from src.llm.prompts import (
    SYSTEM_MEDICO,
    build_diagnosis_prompt,
    build_optimization_report_prompt,
    build_qa_prompt,
)


def interpret_diagnosis(
    patient_data: dict,
    prediction: str,
    probabilities: dict,
    feature_importance: dict,
) -> str:
    prompt = build_diagnosis_prompt(patient_data, prediction, probabilities, feature_importance)
    return chat(prompt, system=SYSTEM_MEDICO, max_tokens=1500)


def generate_optimization_report(
    baseline_metrics: dict,
    optimized_metrics: dict,
    ga_config: dict,
) -> str:
    prompt = build_optimization_report_prompt(baseline_metrics, optimized_metrics, ga_config)
    return chat(prompt, system=SYSTEM_MEDICO, max_tokens=1500)


def answer_question(
    patient_data: dict,
    prediction: str,
    probabilities: dict,
    question: str,
) -> str:
    prompt = build_qa_prompt(patient_data, prediction, probabilities, question)
    return chat(prompt, system=SYSTEM_MEDICO, max_tokens=1024)
