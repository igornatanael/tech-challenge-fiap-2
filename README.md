# Tech Challenge FIAP — Fase 2: Otimização de Modelos de Diagnóstico

Continuação do projeto de suporte ao diagnóstico médico (Fase 1), adicionando **otimização via Algoritmos Genéticos** e **interpretação por LLM (Claude)** aos modelos de classificação de risco gestacional.

## Contexto

Na Fase 1, construímos modelos de classificação de risco gestacional (baixo/médio/alto) a partir de sinais vitais de gestantes — com Random Forest atingindo F1-macro 0.90 e ROC-AUC 0.98.

Nesta fase:
1. **Algoritmos Genéticos** otimizam os hiperparâmetros desses modelos
2. **Claude (Anthropic)** transforma os resultados numéricos em explicações acessíveis para médicos obstetras

## Estrutura do Projeto

```
tech-challenge-fiap-fase2/
├── data/                        # Dataset Maternal Health Risk (UCI)
├── notebooks/
│   ├── 01_baseline_models.ipynb     # Modelos da Fase 1 como baseline
│   ├── 02_genetic_algorithm.ipynb   # AG + experimentos + convergência
│   └── 03_llm_integration.ipynb    # Integração Claude + exemplos
├── src/
│   ├── models/                  # Baseline e modelos otimizados
│   ├── genetic_algorithm/       # Codificação, operadores, fitness, loop AG
│   ├── llm/                     # Cliente Claude, prompts, interpretador
│   └── evaluation/              # Métricas e comparativos
├── tests/                       # Testes automatizados
├── experiments/                 # Resultados dos 3 experimentos AG (JSON)
└── reports/                     # Relatório técnico e figuras
```

## Requisitos

- Python 3.11+
- Chave de API Anthropic ([console.anthropic.com](https://console.anthropic.com/))

## Instalação

```bash
# Clonar repositório
git clone <repo-url>
cd tech-challenge-fiap-fase2

# Criar ambiente virtual
python -m venv .venv
source .venv/bin/activate  # Linux/Mac
# .venv\Scripts\activate   # Windows

# Instalar dependências
pip install -r requirements.txt

# Configurar variáveis de ambiente
cp .env.example .env
# Editar .env e adicionar sua ANTHROPIC_API_KEY
```

## Execução via Docker

```bash
docker build -t tc-fiap-fase2 .
docker run --env-file .env tc-fiap-fase2
```

## Execução dos Notebooks

```bash
jupyter notebook notebooks/
```

Ordem recomendada:
1. `01_baseline_models.ipynb` — reproduz e registra os resultados da Fase 1
2. `02_genetic_algorithm.ipynb` — roda os 3 experimentos do AG
3. `03_llm_integration.ipynb` — demonstra as explicações geradas pelo Claude

## Testes

```bash
pytest tests/ -v --cov=src
```

## Resultados (resumo)

> _Tabela será preenchida após os experimentos._

| Modelo | Baseline F1-macro | AG Exp.1 | AG Exp.2 | AG Exp.3 |
|--------|------------------|----------|----------|----------|
| Random Forest | 0.90 | — | — | — |
| Regressão Logística | — | — | — | — |
| SVM | — | — | — | — |

## Fase 1 (referência)

Repositório da Fase 1: https://github.com/igornatanael/tech-challenge-fiap-fase1
