from __future__ import annotations

SYSTEM_MEDICO = (
    "Voce e um especialista em inteligencia artificial medica assistindo obstetras. "
    "Use linguagem clara e acessivel, sem alarmismo desnecessario. "
    "Evite jargoes tecnicos excessivos — prefira termos que um medico obstetra reconheca facilmente. "
    "Sempre recomende avaliacao medica presencial como etapa essencial, "
    "pois este sistema e uma ferramenta de apoio a decisao, nao um substituto ao julgamento clinico."
)


def build_diagnosis_prompt(
    patient_data: dict,
    prediction: str,
    probabilities: dict,
    feature_importance: dict,
) -> str:
    prob_lines = "\n".join(
        f"  - {classe}: {prob*100:.1f}%" for classe, prob in probabilities.items()
    )
    importance_lines = "\n".join(
        f"  - {feature}: {imp*100:.1f}%" for feature, imp in feature_importance.items()
    )
    data_lines = "\n".join(f"  - {k}: {v}" for k, v in patient_data.items())

    return f"""Analise os dados clinicos de uma paciente gestante e explique a predicao de risco gerada por um modelo de Random Forest treinado no dataset Maternal Health Risk.

## Dados da paciente
{data_lines}

## Predicao do modelo
- Classificacao: **{prediction}**
- Probabilidades por classe:
{prob_lines}

## Importancia das features (contribuicao para a predicao)
{importance_lines}

## O que preciso de voce
1. **Explicacao da predicao**: Em 2-3 paragrafos, explique em linguagem acessivel para um medico obstetra o que esses dados indicam e por que o modelo classificou essa paciente como "{prediction}".
2. **Fatores mais relevantes**: Destaque quais variaveis clinicas mais contribuiram para essa classificacao e o que elas significam neste contexto.
3. **O que monitorar**: Liste os sinais e parametros clinicos que o medico deve acompanhar com maior atencao para esta paciente.
4. **Limitacoes do modelo**: Mencione brevemente o que o modelo NAO consegue capturar e por que a avaliacao presencial e insubstituivel.

Responda de forma estruturada, usando os topicos acima. Seja direto e clinicamente relevante."""


def build_optimization_report_prompt(
    baseline: dict,
    optimized: dict,
    ga_config: dict,
) -> str:
    baseline_lines = "\n".join(f"  - {k}: {v}" for k, v in baseline.items())
    optimized_lines = "\n".join(f"  - {k}: {v}" for k, v in optimized.items())
    config_lines = "\n".join(f"  - {k}: {v}" for k, v in ga_config.items())

    delta_f1 = optimized.get("f1_macro", 0) - baseline.get("f1_macro", 0)
    delta_recall_hr = optimized.get("recall_high_risk", 0) - baseline.get("recall_high_risk", 0)

    return f"""Analise os resultados de otimizacao de hiperparametros de um classificador de risco gestacional realizada por um Algoritmo Genetico (AG).

## Metricas do modelo baseline (Random Forest — GridSearch padrao)
{baseline_lines}

## Metricas do modelo otimizado pelo Algoritmo Genetico
{optimized_lines}

## Configuracao do Algoritmo Genetico utilizado
{config_lines}

## Diferencas absolutas nas metricas-chave
- Delta F1-macro: {delta_f1:+.4f}
- Delta Recall High Risk: {delta_recall_hr:+.4f}

## O que preciso de voce
1. **Analise narrativa da melhoria**: Descreva em linguagem clara o que mudou entre o baseline e o modelo otimizado, interpretando o significado clinico dessas mudancas (nao apenas os numeros).
2. **O que o AG descobriu**: Interprete o que a diferenca de hiperparametros pode revelar sobre a estrutura dos dados — o AG encontrou algo que o GridSearch padrao nao explorou?
3. **Implicacoes clinicas**: Para o contexto de risco gestacional, qual e o impacto pratico da melhoria (ou eventual ausencia de melhoria) nas metricas de recall — especialmente para pacientes de alto risco? Lembre que falsos negativos em alto risco sao clinicamente inaceitaveis.
4. **Recomendacao**: O modelo otimizado deve ser preferido ao baseline? Justifique com base nas metricas clinicamente mais relevantes.

Responda de forma estruturada e objetiva, adequada para um relatorio tecnico-clinico."""


def build_qa_prompt(
    patient_data: dict,
    prediction: str,
    probabilities: dict,
    question: str,
) -> str:
    prob_lines = "\n".join(
        f"  - {classe}: {prob*100:.1f}%" for classe, prob in probabilities.items()
    )
    data_lines = "\n".join(f"  - {k}: {v}" for k, v in patient_data.items())

    return f"""Um medico obstetra tem uma pergunta sobre o caso de uma paciente gestante que foi avaliada por um modelo de IA de classificacao de risco gestacional.

## Dados da paciente
{data_lines}

## Predicao do modelo
- Classificacao: **{prediction}**
- Probabilidades por classe:
{prob_lines}

## Pergunta do medico
{question}

Responda de forma direta, clinicamente relevante e baseada nos dados apresentados. Seja objetivo — o medico precisa de informacao util para tomar decisoes. Se a pergunta extrapolar o que os dados permitem responder, diga isso claramente."""
