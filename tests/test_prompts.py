"""
Testes para src/llm/prompts.py

Verifica apenas a construção das strings de prompt — sem chamadas à API Claude.
"""

from src.llm.prompts import (
    build_diagnosis_prompt,
    build_optimization_report_prompt,
    build_qa_prompt,
)

# ---------------------------------------------------------------------------
# Dados de apoio reutilizados nos testes
# ---------------------------------------------------------------------------

PATIENT_DATA = {
    "Age": 28,
    "SystolicBP": 130,
    "DiastolicBP": 85,
    "BS": 9.5,
    "BodyTemp": 37.2,
    "HeartRate": 88,
}

PROBABILITIES = {"low risk": 0.10, "mid risk": 0.30, "high risk": 0.60}

FEATURE_IMPORTANCE = {
    "SystolicBP": 0.35,
    "BS": 0.25,
    "Age": 0.15,
    "DiastolicBP": 0.10,
    "HeartRate": 0.10,
    "BodyTemp": 0.05,
}

PREDICTION = "high risk"


# ---------------------------------------------------------------------------
# build_diagnosis_prompt
# ---------------------------------------------------------------------------


def test_build_diagnosis_prompt_contains_prediction():
    """O prompt contém a string da predição."""
    prompt = build_diagnosis_prompt(
        patient_data=PATIENT_DATA,
        prediction=PREDICTION,
        probabilities=PROBABILITIES,
        feature_importance=FEATURE_IMPORTANCE,
    )
    assert PREDICTION in prompt


def test_build_diagnosis_prompt_contains_patient_fields():
    """O prompt contém os valores dos campos do paciente."""
    prompt = build_diagnosis_prompt(
        patient_data=PATIENT_DATA,
        prediction=PREDICTION,
        probabilities=PROBABILITIES,
        feature_importance=FEATURE_IMPORTANCE,
    )
    for field, value in PATIENT_DATA.items():
        assert str(value) in prompt, f"Campo '{field}' com valor '{value}' não encontrado no prompt"


# ---------------------------------------------------------------------------
# build_optimization_report_prompt
# ---------------------------------------------------------------------------


def test_build_optimization_report_prompt_contains_metrics():
    """O prompt contém f1_macro do baseline e do modelo otimizado."""
    baseline = {"f1_macro": 0.82, "recall_high_risk": 0.75, "accuracy": 0.80}
    optimized = {"f1_macro": 0.88, "recall_high_risk": 0.83, "accuracy": 0.86}
    ga_config = {
        "model_name": "random_forest",
        "population_size": 30,
        "generations": 20,
        "mutation_rate": 0.1,
        "crossover_rate": 0.8,
    }

    prompt = build_optimization_report_prompt(
        baseline=baseline,
        optimized=optimized,
        ga_config=ga_config,
    )

    # Ambos os valores f1_macro devem aparecer no prompt
    assert str(baseline["f1_macro"]) in prompt
    assert str(optimized["f1_macro"]) in prompt


# ---------------------------------------------------------------------------
# build_qa_prompt
# ---------------------------------------------------------------------------


def test_build_qa_prompt_contains_question():
    """O prompt contém a pergunta do médico."""
    question = "Quais são os riscos de pré-eclâmpsia para esta paciente?"

    prompt = build_qa_prompt(
        patient_data=PATIENT_DATA,
        prediction=PREDICTION,
        probabilities=PROBABILITIES,
        question=question,
    )

    assert question in prompt
