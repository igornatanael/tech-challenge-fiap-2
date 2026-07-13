"""
App Streamlit — Chatbot de Avaliação de Risco Gestacional
Tech Challenge FIAP — Fase 2

Execução:
    streamlit run app.py
"""

from __future__ import annotations

import sys
import uuid
from pathlib import Path

import matplotlib
matplotlib.use("Agg")

import pandas as pd
import streamlit as st
from dotenv import load_dotenv

load_dotenv()

PROJECT_ROOT = Path(__file__).resolve().parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.pipelines.preprocessing import build_preprocessed_splits, RISK_DECODING
from src.models.baseline import build_pipelines
from src.llm.agents import PatientAgent, DoctorAgent
from src.observability import setup_logging, log_event

setup_logging()

# ---------------------------------------------------------------------------
# Configuração da página
# ---------------------------------------------------------------------------

st.set_page_config(
    page_title="Assistente de Risco Gestacional",
    page_icon="🤰",
    layout="centered",
)

# ---------------------------------------------------------------------------
# Campos de coleta
# ---------------------------------------------------------------------------

FIELDS = [
    {
        "key": "Age",
        "question": "Qual é a **idade** da paciente? (em anos)",
        "unit": "anos",
        "type": int,
        "min": 10,
        "max": 70,
        "error": "A idade deve estar entre 10 e 70 anos. Por favor, insira um valor válido.",
    },
    {
        "key": "SystolicBP",
        "question": "Qual é a **pressão sistólica** (o número de cima)? (em mmHg)",
        "unit": "mmHg",
        "type": int,
        "min": 60,
        "max": 250,
        "error": "A pressão sistólica deve estar entre 60 e 250 mmHg.",
    },
    {
        "key": "DiastolicBP",
        "question": "Qual é a **pressão diastólica** (o número de baixo)? (em mmHg)",
        "unit": "mmHg",
        "type": int,
        "min": 40,
        "max": 150,
        "error": "A pressão diastólica deve estar entre 40 e 150 mmHg.",
        "cross_validate_key": "SystolicBP",
        "cross_error": "A pressão diastólica deve ser menor que a sistólica ({SystolicBP} mmHg). Verifique os valores.",
    },
    {
        "key": "BS",
        "question": "Qual é a **glicemia** da paciente? (em mmol/L)",
        "unit": "mmol/L",
        "type": float,
        "min": 1.0,
        "max": 30.0,
        "error": "A glicemia deve estar entre 1.0 e 30.0 mmol/L.",
    },
    {
        "key": "BodyTemp",
        "question": "Qual é a **temperatura corporal**? (em °C)",
        "unit": "°C",
        "type": float,
        "min": 35.0,
        "max": 41.0,
        "error": "A temperatura deve estar entre 35.0 e 41.0 °C.",
    },
    {
        "key": "HeartRate",
        "question": "Qual é a **frequência cardíaca**? (em bpm)",
        "unit": "bpm",
        "type": int,
        "min": 30,
        "max": 200,
        "error": "A frequência cardíaca deve estar entre 30 e 200 bpm.",
    },
]

RESET_KEYWORDS = {"nova avaliação", "nova avaliacao", "recomeçar", "recomecar", "reiniciar"}

# ---------------------------------------------------------------------------
# Cache — treina o modelo uma vez
# ---------------------------------------------------------------------------

@st.cache_resource(show_spinner="Carregando modelo...")
def load_model():
    X_train, _, y_train, _, _ = build_preprocessed_splits()
    pipeline = build_pipelines()["random_forest"]
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

# ---------------------------------------------------------------------------
# Gerenciamento de estado
# ---------------------------------------------------------------------------

def init_session():
    defaults = {
        "started": False,
        "messages": [],
        "state": "ask_role",
        "role": None,
        "patient_data": {},
        "field_index": 0,
        "agent": None,
        "diagnosed": False,
        "session_id": str(uuid.uuid4()),
    }
    for key, value in defaults.items():
        if key not in st.session_state:
            st.session_state[key] = value


def reset_session():
    keys_to_clear = [
        "started", "messages", "state", "role",
        "patient_data", "field_index", "agent", "diagnosed",
    ]
    for key in keys_to_clear:
        if key in st.session_state:
            del st.session_state[key]
    st.rerun()


def add_message(role: str, content: str):
    st.session_state.messages.append({"role": role, "content": content})

# ---------------------------------------------------------------------------
# Diagnóstico
# ---------------------------------------------------------------------------

def run_diagnosis():
    pipeline, feature_importance = load_model()
    patient_data = st.session_state.patient_data.copy()

    # Modelo foi treinado com temperatura em °F — converter de °C
    patient_data["BodyTemp"] = round(patient_data["BodyTemp"] * 9 / 5 + 32, 1)

    X = pd.DataFrame([patient_data])
    pred_idx = int(pipeline.predict(X)[0])
    probas = pipeline.predict_proba(X)[0]

    risk_level = RISK_DECODING[pred_idx]
    probabilities = {RISK_DECODING[i]: round(float(p), 4) for i, p in enumerate(probas)}

    log_event("model.prediction", session_id=st.session_state.session_id,
              risk_level=risk_level, probabilities=probabilities)

    role = st.session_state.role
    agent = PatientAgent() if role == "patient" else DoctorAgent()
    agent.session_id = st.session_state.session_id
    st.session_state.agent = agent

    response = agent.iniciar_consulta(
        patient_data=patient_data,
        risk_level=risk_level,
        probabilities=probabilities,
        feature_importance=feature_importance,
    )

    add_message("assistant", response)
    st.session_state.diagnosed = True
    st.session_state.state = "qa"

# ---------------------------------------------------------------------------
# Processamento de input
# ---------------------------------------------------------------------------

def process_input(user_input: str):
    text = user_input.strip()

    if text.lower() in RESET_KEYWORDS:
        reset_session()
        return

    add_message("user", text)
    state = st.session_state.state

    if state == "ask_role":
        normalized = text.lower().strip()
        if normalized in ("médico", "medico", "m", "doctor", "dr", "dra"):
            st.session_state.role = "doctor"
            log_event("session.started", session_id=st.session_state.session_id, role="doctor")
            confirmation = "Certo! Vou fornecer uma análise clínica detalhada. Vamos começar coletando os dados da paciente.\n\n"
            confirmation += FIELDS[0]["question"]
            add_message("assistant", confirmation)
            st.session_state.state = "collecting"
        elif normalized in ("paciente", "p", "patient", "gestante"):
            st.session_state.role = "patient"
            log_event("session.started", session_id=st.session_state.session_id, role="patient")
            confirmation = "Olá! Vou te ajudar a entender melhor sua saúde na gestação. Vamos precisar de algumas informações.\n\n"
            confirmation += FIELDS[0]["question"]
            add_message("assistant", confirmation)
            st.session_state.state = "collecting"
        else:
            add_message(
                "assistant",
                "Não entendi. Por favor, responda **médico** ou **paciente** para que eu possa adaptar a explicação para você.",
            )

    elif state == "collecting":
        field = FIELDS[st.session_state.field_index]
        field_type = field["type"]

        try:
            value = field_type(text.replace(",", "."))
        except (ValueError, TypeError):
            tipo = "número inteiro" if field_type is int else "número"
            add_message("assistant", f"Preciso de um {tipo}. {field['error']}")
            return

        if not (field["min"] <= value <= field["max"]):
            add_message("assistant", field["error"])
            return

        if "cross_validate_key" in field:
            ref_key = field["cross_validate_key"]
            ref_value = st.session_state.patient_data.get(ref_key)
            if ref_value is not None and value >= ref_value:
                error_msg = field["cross_error"].format(**{ref_key: ref_value})
                add_message("assistant", error_msg)
                return

        st.session_state.patient_data[field["key"]] = value
        st.session_state.field_index += 1

        if st.session_state.field_index < len(FIELDS):
            next_field = FIELDS[st.session_state.field_index]
            add_message("assistant", next_field["question"])
        else:
            log_event("data.collected", session_id=st.session_state.session_id,
                      fields=list(st.session_state.patient_data.keys()))
            add_message("assistant", "Perfeito! Tenho todos os dados. Analisando agora...")
            st.session_state.state = "diagnosing"

    elif state == "qa":
        agent = st.session_state.agent
        if agent is None:
            add_message("assistant", "Ocorreu um erro. Por favor, inicie uma nova avaliação.")
            return
        log_event("qa.question", session_id=st.session_state.session_id,
                  turn=len(agent.history) // 2 + 1)
        response = agent.chat(text)
        add_message("assistant", response)

# ---------------------------------------------------------------------------
# Tela de landing
# ---------------------------------------------------------------------------

def render_landing():
    st.title("Assistente de Risco Gestacional")
    st.markdown(
        "Avaliação de risco gestacional baseada em dados clínicos, com explicações "
        "geradas por inteligência artificial.\n\n"
        "O assistente coleta os dados da paciente, classifica o risco com um modelo "
        "de Random Forest otimizado e fornece orientações adaptadas ao perfil do usuário — "
        "médico ou paciente."
    )
    st.warning(
        "Esta ferramenta é um apoio à decisão. **Não substitui avaliação médica presencial.**",
        icon="⚠️",
    )
    st.markdown("<br>", unsafe_allow_html=True)

    col_left, col_center, col_right = st.columns([1, 2, 1])
    with col_center:
        if st.button("Iniciar Avaliação", type="primary", use_container_width=True):
            st.session_state.started = True
            st.rerun()

# ---------------------------------------------------------------------------
# Tela de chatbot
# ---------------------------------------------------------------------------

def render_chat():
    col_title, col_btn = st.columns([4, 1])
    with col_title:
        st.markdown("### Assistente de Risco Gestacional")
    with col_btn:
        if st.button("Nova Avaliação", use_container_width=True):
            reset_session()

    st.divider()

    for msg in st.session_state.messages:
        with st.chat_message(msg["role"]):
            st.markdown(msg["content"])

    if st.session_state.state == "diagnosing":
        with st.spinner("Analisando dados e gerando avaliação..."):
            run_diagnosis()
        st.rerun()

    user_input = st.chat_input("Digite sua resposta...")

    if user_input:
        process_input(user_input)
        st.rerun()

# ---------------------------------------------------------------------------
# Inicialização da primeira mensagem
# ---------------------------------------------------------------------------

def maybe_greet():
    if not st.session_state.messages:
        greeting = (
            "Olá! Sou o Assistente de Risco Gestacional.\n\n"
            "Vou te ajudar a avaliar o risco gestacional com base em alguns dados clínicos.\n\n"
            "Antes de começar: você é **médico** ou **paciente**?"
        )
        add_message("assistant", greeting)

# ---------------------------------------------------------------------------
# Ponto de entrada
# ---------------------------------------------------------------------------

init_session()

if not st.session_state.started:
    render_landing()
else:
    maybe_greet()
    render_chat()

st.divider()
st.caption(
    "Tech Challenge FIAP PosTech — Fase 2 · "
    "Dataset: Maternal Health Risk (UCI) · "
    "Modelo: Random Forest otimizado por Algoritmo Genético · "
    "LLM: Claude Sonnet 4.6 (Anthropic)"
)
