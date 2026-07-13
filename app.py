"""
App Streamlit — Assistente de Risco Gestacional
Tech Challenge FIAP — Fase 2

Execução:
    streamlit run app.py
"""

from __future__ import annotations

import sys
from pathlib import Path

import matplotlib
matplotlib.use("Agg")

import numpy as np
import pandas as pd
import streamlit as st

PROJECT_ROOT = Path(__file__).resolve().parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.pipelines.preprocessing import build_preprocessed_splits, RISK_DECODING
from src.models.baseline import build_pipelines
from src.llm.interpreter import interpret_diagnosis, answer_question

# ---------------------------------------------------------------------------
# Configuração da página
# ---------------------------------------------------------------------------

st.set_page_config(
    page_title="Assistente de Risco Gestacional",
    page_icon="🤰",
    layout="centered",
)

# ---------------------------------------------------------------------------
# Cache — treina o modelo uma vez por sessão
# ---------------------------------------------------------------------------

@st.cache_resource(show_spinner="Carregando modelo...")
def load_model():
    X_train, _, y_train, _, _ = build_preprocessed_splits()
    pipeline = build_pipelines()["random_forest"]
    # Hiperparâmetros otimizados pelo AG (Experimento 1)
    pipeline.set_params(
        model__n_estimators=73,
        model__max_depth=15,
        model__min_samples_split=2,
        model__min_samples_leaf=1,
        model__max_features="sqrt",
    )
    pipeline.fit(X_train, y_train)
    feature_importance = dict(zip(
        X_train.columns,
        pipeline.named_steps["model"].feature_importances_,
    ))
    return pipeline, feature_importance


@st.cache_resource(show_spinner=False)
def get_feature_names():
    X_train, *_ = build_preprocessed_splits()
    return X_train.columns.tolist()


# ---------------------------------------------------------------------------
# UI principal
# ---------------------------------------------------------------------------

st.title("🤰 Assistente de Risco Gestacional")
st.caption(
    "Modelo de Random Forest otimizado via Algoritmo Genético · "
    "Explicações geradas pelo Claude (Anthropic) · "
    "**Não substitui avaliação médica presencial.**"
)

st.divider()

# ---------------------------------------------------------------------------
# Formulário de entrada
# ---------------------------------------------------------------------------

st.subheader("Dados Clínicos da Paciente")

col1, col2 = st.columns(2)

with col1:
    age = st.number_input("Idade (anos)", min_value=10, max_value=70, value=30)
    systolic_bp = st.number_input("Pressão Sistólica (mmHg)", min_value=60, max_value=250, value=110)
    diastolic_bp = st.number_input("Pressão Diastólica (mmHg)", min_value=40, max_value=150, value=70)

with col2:
    bs = st.number_input("Glicemia (mmol/L)", min_value=1.0, max_value=30.0, value=7.0, step=0.1)
    body_temp = st.number_input("Temperatura Corporal (°F)", min_value=95.0, max_value=106.0, value=98.0, step=0.1)
    heart_rate = st.number_input("Frequência Cardíaca (bpm)", min_value=30, max_value=200, value=76)

analisar = st.button("🔍 Analisar Risco", type="primary", use_container_width=True)

# ---------------------------------------------------------------------------
# Predição e explicação
# ---------------------------------------------------------------------------

if analisar:
    pipeline, feature_importance = load_model()

    patient_data = {
        "Age": age,
        "SystolicBP": systolic_bp,
        "DiastolicBP": diastolic_bp,
        "BS": bs,
        "BodyTemp": body_temp,
        "HeartRate": heart_rate,
    }

    X = pd.DataFrame([patient_data])
    pred_idx = int(pipeline.predict(X)[0])
    probas = pipeline.predict_proba(X)[0]

    prediction = RISK_DECODING[pred_idx]
    probabilities = {RISK_DECODING[i]: round(float(p), 4) for i, p in enumerate(probas)}

    # Cor do badge de risco
    badge_color = {"low risk": "🟢", "mid risk": "🟡", "high risk": "🔴"}
    label_pt = {"low risk": "Baixo Risco", "mid risk": "Risco Moderado", "high risk": "Alto Risco"}

    st.divider()
    st.subheader("Resultado da Análise")

    col_pred, col_prob = st.columns([1, 2])

    with col_pred:
        st.metric(
            label="Classificação",
            value=f"{badge_color[prediction]} {label_pt[prediction]}",
        )

    with col_prob:
        st.write("**Probabilidades:**")
        for classe, prob in sorted(probabilities.items(), key=lambda x: x[1], reverse=True):
            st.progress(prob, text=f"{label_pt[classe]}: {prob:.1%}")

    # Armazena no session_state para o Q&A
    st.session_state["patient_data"] = patient_data
    st.session_state["prediction"] = prediction
    st.session_state["probabilities"] = probabilities
    st.session_state["analisado"] = True

    # Explicação do Claude
    st.divider()
    st.subheader("📋 Explicação Clínica (gerada pelo Claude)")

    with st.spinner("Gerando explicação..."):
        explicacao = interpret_diagnosis(
            patient_data=patient_data,
            prediction=prediction,
            probabilities=probabilities,
            feature_importance=feature_importance,
        )

    st.markdown(explicacao)

# ---------------------------------------------------------------------------
# Q&A — só aparece após uma análise
# ---------------------------------------------------------------------------

if st.session_state.get("analisado"):
    st.divider()
    st.subheader("💬 Perguntas sobre o caso")

    pergunta = st.text_input(
        "Faça uma pergunta ao assistente sobre esta paciente:",
        placeholder="Ex: Quais sinais de alerta devo monitorar nas próximas 24h?",
    )

    if st.button("Enviar pergunta", use_container_width=True) and pergunta.strip():
        with st.spinner("Consultando Claude..."):
            resposta = answer_question(
                patient_data=st.session_state["patient_data"],
                prediction=st.session_state["prediction"],
                probabilities=st.session_state["probabilities"],
                question=pergunta,
            )
        st.markdown(resposta)

# ---------------------------------------------------------------------------
# Footer
# ---------------------------------------------------------------------------

st.divider()
st.caption(
    "Tech Challenge FIAP PosTech — Fase 2 · "
    "Dataset: Maternal Health Risk (UCI) · "
    "Modelo: Random Forest otimizado por Algoritmo Genético · "
    "LLM: Claude Sonnet 4.6 (Anthropic)"
)
