from __future__ import annotations

from src.llm.agents.base_agent import BaseAgent


class DoctorAgent(BaseAgent):
    SYSTEM_PROMPT = """Você é um assistente de IA especializado em saúde materna, desenvolvido para auxiliar médicos obstetras na avaliação de risco gestacional.

Tom: técnico, direto e baseado em evidências. Use terminologia clínica adequada.

Você pode:
- Fornecer análise clínica detalhada de dados obstétricos
- Sugerir investigação complementar e conduta baseada nos achados do caso
- Interpretar resultados do modelo de predição de risco gestacional com contexto clínico
- Responder perguntas sobre avaliação clínica obstétrica e saúde materna

Valores de referência relevantes:
- Pressão arterial: normal <120/80, pré-hipertensão 120-139/80-89, HAS ≥140/90, crise hipertensiva ≥180/120
- Glicemia (mmol/L): normal em jejum <5.6, pré-diabetes 5.6-6.9, diabetes ≥7.0
- Temperatura (°F): normal 97.8-99.1, febril ≥100.4
- Frequência cardíaca: normal 60-100 bpm, taquicardia >100

Você NÃO responde perguntas administrativas, jurídicas, pessoais ou completamente fora do escopo de avaliação clínica obstétrica.

Ao receber uma pergunta fora do escopo, responda exatamente: "Esta questão está fora do escopo de suporte clínico obstétrico. Para outras questões, consulte os canais apropriados."

Suas análises são suporte à decisão clínica — a responsabilidade diagnóstica e terapêutica é do médico assistente."""

    def __init__(self):
        super().__init__(self.SYSTEM_PROMPT, max_tokens=1800)

    def iniciar_consulta(
        self,
        patient_data: dict,
        risk_level: str,
        probabilities: dict,
        feature_importance: dict,
    ) -> str:
        prompt = _build_doctor_initial_prompt(patient_data, risk_level, probabilities, feature_importance)
        return self.chat(prompt)


def _build_doctor_initial_prompt(
    patient_data: dict,
    risk_level: str,
    probabilities: dict,
    feature_importance: dict,
) -> str:
    top_features = sorted(feature_importance.items(), key=lambda x: x[1], reverse=True)[:4]
    top_features_lines = "\n".join(
        f"  - {feat}: importância {imp:.4f} | valor da paciente: {patient_data.get(feat, 'N/A')}"
        for feat, imp in top_features
    )

    probabilities_lines = "\n".join(
        f"  - {cls}: {prob:.1%}"
        for cls, prob in sorted(probabilities.items(), key=lambda x: x[1], reverse=True)
    )

    age = patient_data.get("Age", "N/A")
    systolic = patient_data.get("SystolicBP", "N/A")
    diastolic = patient_data.get("DiastolicBP", "N/A")
    bs = patient_data.get("BS", "N/A")
    body_temp = patient_data.get("BodyTemp", "N/A")
    heart_rate = patient_data.get("HeartRate", "N/A")

    return f"""Dados clínicos da paciente para análise:

**Dados vitais e laboratoriais:**
- Idade: {age} anos
- PA: {systolic}/{diastolic} mmHg
- Glicemia: {bs} mmol/L
- Temperatura: {body_temp} °F
- FC: {heart_rate} bpm

**Output do modelo de predição (Random Forest otimizado por AG):**
- Predição: {risk_level}
- Distribuição de probabilidades:
{probabilities_lines}

**Feature importance (top 4 features do modelo):**
{top_features_lines}

Por favor, estruture sua análise nas seguintes seções:

## 1. Análise do Modelo
Interprete a predição e o nível de confiança. Comente quais features mais influenciaram o resultado e o que os valores individuais desta paciente indicam no contexto do modelo.

## 2. Avaliação Clínica
Interprete os achados individuais com base nos valores de referência clínicos. Identifique combinações de risco relevantes (ex: PA elevada + idade avançada → suspeita de pré-eclâmpsia; glicemia elevada → DMG; etc.). Seja específico para este caso — não use análise genérica por classe de risco.

## 3. Investigação Sugerida
Liste os exames complementares pertinentes para este caso específico, justificando cada um com base nos achados clínicos.

## 4. Conduta Recomendada
Indique os próximos passos clínicos com base nos achados (internação, ambulatório urgente, pré-natal de rotina, encaminhamento especializado, etc.). Inclua nível de urgência.

Ao final, coloque-se à disposição para perguntas sobre o caso."""
