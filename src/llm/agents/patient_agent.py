from __future__ import annotations

from src.llm.agents.base_agent import BaseAgent


class PatientAgent(BaseAgent):
    SYSTEM_PROMPT = """Você é um assistente de saúde que ajuda gestantes a entender seu risco gestacional.

Tom: acolhedor, empático, claro e sem alarmismo desnecessário. Fale como se estivesse explicando para uma amiga próxima.

Linguagem: simples, evite termos médicos complexos. Se precisar usar um termo técnico, explique-o em seguida.

Você pode:
- Explicar resultados de avaliação de risco gestacional em linguagem acessível
- Dar orientações práticas sobre sinais de alerta e cuidados gerais na gestação
- Responder dúvidas sobre saúde na gestação

Você NÃO pode:
- Prescrever medicamentos ou alterar prescrições existentes
- Substituir uma consulta médica presencial
- Responder perguntas completamente fora do contexto de saúde gestacional (culinária, política, tecnologia, etc.)

Ao receber uma pergunta fora do escopo, responda exatamente: "Posso te ajudar com dúvidas sobre saúde na gestação. Para outras perguntas, consulte os recursos apropriados."

Lembre sempre que você é um apoio à decisão — a avaliação médica presencial é insubstituível."""

    def __init__(self):
        super().__init__(self.SYSTEM_PROMPT, max_tokens=1200)

    def iniciar_consulta(
        self,
        patient_data: dict,
        risk_level: str,
        probabilities: dict,
        feature_importance: dict,
    ) -> str:
        prompt = _build_patient_initial_prompt(patient_data, risk_level, probabilities, feature_importance)
        return self.chat(prompt)


def _build_patient_initial_prompt(
    patient_data: dict,
    risk_level: str,
    probabilities: dict,
    feature_importance: dict,
) -> str:
    risk_labels = {
        "low risk": "baixo risco",
        "mid risk": "risco moderado",
        "high risk": "alto risco",
    }
    risk_label_pt = risk_labels.get(risk_level, risk_level)

    top_features = sorted(feature_importance.items(), key=lambda x: x[1], reverse=True)[:3]
    top_features_str = ", ".join(f"{k} (valor: {patient_data.get(k, 'N/A')})" for k, _ in top_features)

    age = patient_data.get("Age", "N/A")
    systolic = patient_data.get("SystolicBP", "N/A")
    diastolic = patient_data.get("DiastolicBP", "N/A")
    bs = patient_data.get("BS", "N/A")
    body_temp = patient_data.get("BodyTemp", "N/A")
    heart_rate = patient_data.get("HeartRate", "N/A")

    return f"""Você está avaliando uma gestante com os seguintes dados clínicos:
- Idade: {age} anos
- Pressão arterial: {systolic}/{diastolic} mmHg
- Glicemia: {bs} mmol/L
- Temperatura corporal: {body_temp} °F
- Frequência cardíaca: {heart_rate} bpm

O modelo de avaliação identificou: **{risk_label_pt}**.

Os fatores que mais contribuíram para essa avaliação foram: {top_features_str}.

Por favor, elabore uma resposta para a paciente seguindo EXATAMENTE esta estrutura:

1. **Como está sua saúde**: Explique o resultado em linguagem simples e acolhedora. NÃO diga "probabilidade de X%". Em vez disso, diga algo como "o modelo identificou sinais que merecem atenção" ou "seus dados mostram que você está bem no momento". Adapte ao nível de risco real.

2. **O que isso significa para você**: Com base nos dados individuais desta paciente (não de forma genérica), dê orientações práticas e específicas:
   - Se a glicemia ({bs} mmol/L) estiver elevada (acima de 7.0): mencione evitar alimentos açucarados e refinados
   - Se a pressão sistólica ({systolic} mmHg) estiver alta (acima de 140): oriente descanso, evitar esforço e buscar atendimento hoje
   - Se a frequência cardíaca ({heart_rate} bpm) estiver elevada (acima de 100): oriente evitar atividade física intensa
   - Se a temperatura ({body_temp} °F) estiver elevada (acima de 99.5): mencione hidratação e contato com médico
   - Adapte conforme os valores reais — se estiverem normais, tranquilize a paciente nesse aspecto

3. **O que fazer agora**: Oriente claramente uma das três opções com base no risco real:
   - Alto risco ({risk_level == "high risk"}): "Busque atendimento de emergência hoje"
   - Risco moderado ({risk_level == "mid risk"}): "Marque uma consulta com seu médico com urgência nos próximos dias"
   - Baixo risco ({risk_level == "low risk"}): "Continue com suas consultas de pré-natal de rotina"

4. Encerre perguntando se ela tem alguma dúvida sobre o resultado ou sobre sua saúde na gestação.

Mantenha a resposta calorosa, clara e prática. Não use jargões médicos sem explicação."""
