from __future__ import annotations

import json
import re

from src.llm.client import _get_client, DEFAULT_MODEL

_RUBRIC_PATIENT = """
Avalie a resposta do assistente para a paciente gestante usando os seguintes critérios (nota de 1 a 5 cada):

- clareza: linguagem acessível, sem jargão médico não explicado (1=muito técnico, 5=totalmente acessível)
- tom_adequado: acolhedor, sem alarmismo excessivo nem minimização indevida (1=inadequado, 5=muito adequado)
- urgencia_correta: a urgência recomendada condiz com o nível de risco real (1=completamente errada, 5=perfeitamente adequada)
- acionabilidade: recomendações práticas e específicas para os dados desta paciente, não genéricas (1=vaga/genérica, 5=muito específica e acionável)

Além disso, avalie:
- dentro_escopo: a resposta não prescreveu medicamentos e não extrapolou o escopo de saúde gestacional? (true/false)

Responda APENAS com um JSON válido no seguinte formato:
{
  "scores": {
    "clareza": <int 1-5>,
    "tom_adequado": <int 1-5>,
    "urgencia_correta": <int 1-5>,
    "acionabilidade": <int 1-5>
  },
  "within_scope": <true|false>,
  "justificativa": "<string explicando brevemente cada nota e o veredicto de escopo>"
}"""

_RUBRIC_DOCTOR = """
Avalie a resposta do assistente ao médico obstetra usando os seguintes critérios (nota de 1 a 5 cada):

- precisao_clinica: os achados clínicos interpretados condizem com os dados apresentados e com a literatura (1=impreciso, 5=muito preciso)
- completude: a resposta cobre os aspectos relevantes do caso (análise, investigação, conduta) (1=muito incompleta, 5=muito completa)
- acionabilidade: a conduta e a investigação sugeridas são específicas para este caso, não genéricas (1=genérica, 5=muito específica)
- terminologia: uso adequado de terminologia clínica obstétrica (1=inadequado, 5=muito adequado)

Além disso, avalie:
- dentro_escopo: a resposta ficou no contexto obstétrico/clínico sem extrapolar para questões não médicas? (true/false)

Responda APENAS com um JSON válido no seguinte formato:
{
  "scores": {
    "precisao_clinica": <int 1-5>,
    "completude": <int 1-5>,
    "acionabilidade": <int 1-5>,
    "terminologia": <int 1-5>
  },
  "within_scope": <true|false>,
  "justificativa": "<string explicando brevemente cada nota e o veredicto de escopo>"
}"""


def evaluate_response(
    response: str,
    agent_type: str,
    patient_data: dict,
    risk_level: str,
) -> dict:
    """
    Avalia a qualidade de uma resposta usando Claude como juiz.

    Parâmetros
    ----------
    response : str
        Texto da resposta a ser avaliada.
    agent_type : str
        "patient" ou "doctor".
    patient_data : dict
        Dados clínicos da paciente.
    risk_level : str
        Nível de risco predito ("low risk", "mid risk", "high risk").

    Retorno
    -------
    dict com chaves: scores, within_scope, justificativa, score_total
    """
    rubric = _RUBRIC_PATIENT if agent_type == "patient" else _RUBRIC_DOCTOR

    prompt = f"""Você é um avaliador especialista em qualidade de respostas de assistentes de saúde materna.

Contexto do caso:
- Dados da paciente: {json.dumps(patient_data, ensure_ascii=False)}
- Nível de risco predito: {risk_level}
- Tipo de destinatário da resposta: {"paciente gestante" if agent_type == "patient" else "médico obstetra"}

Resposta a ser avaliada:
\"\"\"
{response}
\"\"\"

{rubric}"""

    client = _get_client()
    api_response = client.messages.create(
        model=DEFAULT_MODEL,
        max_tokens=800,
        messages=[{"role": "user", "content": prompt}],
    )
    raw = api_response.content[0].text.strip()

    json_match = re.search(r"\{.*\}", raw, re.DOTALL)
    if json_match:
        raw = json_match.group(0)

    result = json.loads(raw)

    scores = result.get("scores", {})
    score_total = round(sum(scores.values()) / len(scores), 2) if scores else 0.0

    return {
        "scores": scores,
        "within_scope": result.get("within_scope", False),
        "justificativa": result.get("justificativa", ""),
        "score_total": score_total,
    }
