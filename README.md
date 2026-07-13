# Tech Challenge FIAP — Fase 2: Otimização de Modelos de Diagnóstico

Continuação do projeto de suporte ao diagnóstico médico (Fase 1), adicionando **otimização via Algoritmos Genéticos** e **interpretação por LLM (Claude)** aos modelos de classificação de risco gestacional.

## Contexto

Na Fase 1, construímos modelos de classificação de risco gestacional (baixo/médio/alto) a partir de sinais vitais de gestantes — com Random Forest atingindo F1-macro 0.90 e ROC-AUC 0.98.

Nesta fase:
1. **Algoritmos Genéticos** otimizam os hiperparâmetros do Random Forest
2. **Claude (Anthropic)** transforma os resultados numéricos em explicações acessíveis para médicos obstetras
3. **App web (Streamlit)** permite ao médico inserir dados clínicos e receber diagnóstico + explicação em tempo real

## Resultados

### Baseline vs. AG — Random Forest (conjunto de teste)

| Configuração | F1-macro | ROC-AUC | Recall Alto Risco | Acurácia |
|---|---|---|---|---|
| Baseline (GridSearch) | 0.9017 | 0.9813 | 0.9487 | 0.8987 |
| **AG Exp.1 — Padrão** (pop=30, gen=20, mut=0.10) | **0.9070** | 0.9811 | **0.9744** | **0.9051** |
| AG Exp.2 — Alta Mutação (pop=30, gen=20, mut=0.30) | 0.9017 | — | 0.9487 | — |
| AG Exp.3 — Pop. Grande (pop=60, gen=30, mut=0.10) | 0.9013 | — | 0.9744 | — |

O Experimento 1 superou o baseline em F1-macro (+0.0053) e recall de alto risco (+0.0257), identificando que árvores menores (`max_depth=15`, `n_estimators=73`) generalizam melhor que o GridSearch default.

### Melhores hiperparâmetros encontrados (Exp.1)

| Hiperparâmetro | GridSearch | AG Exp.1 |
|---|---|---|
| `n_estimators` | 100 | 73 |
| `max_depth` | None | 15 |
| `min_samples_split` | 2 | 2 |
| `min_samples_leaf` | 1 | 1 |
| `max_features` | — | sqrt |

## Estrutura do Projeto

```
tech-challenge-fiap-fase2/
├── data/                            # Dataset Maternal Health Risk (UCI)
├── notebooks/
│   ├── 01_baseline_models.ipynb     # Modelos da Fase 1 como baseline
│   ├── 02_genetic_algorithm.ipynb   # AG + 3 experimentos + curvas de convergência
│   └── 03_llm_integration.ipynb    # Integração Claude + exemplos de diagnóstico
├── src/
│   ├── models/baseline.py           # Pipelines sklearn + GridSearch
│   ├── pipelines/                   # Pré-processamento e carregamento de dados
│   ├── genetic_algorithm/           # Codificação, operadores, fitness, loop AG
│   │   ├── encoding.py              # SEARCH_SPACES, random_individual, decode
│   │   ├── operators.py             # Torneio, cruzamento uniforme, mutação
│   │   ├── fitness.py               # StratifiedKFold CV como função fitness
│   │   └── ga.py                    # Loop principal com elitismo
│   ├── llm/                         # Integração Claude (Anthropic)
│   │   ├── client.py                # Wrapper do SDK Anthropic
│   │   ├── prompts.py               # Templates de prompt (contexto médico)
│   │   └── interpreter.py           # interpret_diagnosis, generate_optimization_report, answer_question
│   └── evaluation/                  # Métricas, matrizes de confusão, feature importance
├── tests/                           # 21 testes automatizados (pytest)
├── experiments/                     # Resultados dos 3 experimentos AG (JSON)
├── reports/figures/                 # Gráficos gerados pelos notebooks
├── app.py                           # App web Streamlit
└── Dockerfile                       # Container para testes/execução headless
```

## Requisitos

- Python 3.11+
- Chave de API Anthropic ([console.anthropic.com](https://console.anthropic.com/))

## Instalação

```bash
git clone <repo-url>
cd tech-challenge-fiap-fase2

python -m venv .venv
source .venv/bin/activate       # Linux/Mac
# .venv\Scripts\activate        # Windows

pip install -r requirements.txt

cp .env.example .env
# Editar .env e adicionar: ANTHROPIC_API_KEY=sk-ant-...
```

## App Web

```bash
streamlit run app.py
```

Acesse `http://localhost:8501`. O médico preenche os 6 sinais vitais da paciente e recebe:
- Classificação de risco (baixo / moderado / alto) com probabilidades
- Explicação clínica gerada pelo Claude
- Campo de perguntas livres sobre o caso (Q&A)

## Execução dos Notebooks

```bash
jupyter notebook notebooks/
```

Ordem recomendada:
1. `01_baseline_models.ipynb` — reproduz e registra os resultados da Fase 1
2. `02_genetic_algorithm.ipynb` — roda os 3 experimentos do AG (~40 min)
3. `03_llm_integration.ipynb` — demonstra as explicações geradas pelo Claude

Ou via linha de comando (headless):

```bash
MPLBACKEND=Agg jupyter nbconvert --to notebook --execute --inplace \
  --ExecutePreprocessor.timeout=3600 notebooks/02_genetic_algorithm.ipynb
```

## Testes

```bash
pytest tests/ -v --tb=short
```

21 testes cobrindo: codificação genética, operadores (seleção, cruzamento, mutação), loop do AG (reprodutibilidade, monotonicicidade do fitness) e construção de prompts LLM.

## Docker

```bash
docker build -t tc-fiap-fase2 .

# Roda os testes dentro do container
docker run tc-fiap-fase2

# Passa a chave da API via variável de ambiente (nunca no Dockerfile)
docker run -e ANTHROPIC_API_KEY=sk-ant-... tc-fiap-fase2
```

## Dataset

**Maternal Health Risk Data Set** — UCI Machine Learning Repository  
790 registros, 6 features clínicas: `Age`, `SystolicBP`, `DiastolicBP`, `BS` (glicemia), `BodyTemp`, `HeartRate`  
Classes: `low risk` (0), `mid risk` (1), `high risk` (2)
