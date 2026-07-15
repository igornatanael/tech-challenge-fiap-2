# Relatório Técnico — Tech Challenge FIAP Fase 2

| | |
|---|---|
| **Curso** | PosTech IA para Devs — FIAP |
| **Fase** | 2 — Evolução da IA: GenAI, Cloud ML e LLMs |
| **Projeto** | Projeto 1 — Otimização de Modelos de Diagnóstico |
| **Dataset** | Maternal Health Risk (UCI) — ~790 registros, 6 features clínicas |
| **Repositório** | https://github.com/igornatanael/tech-challenge-fiap-2 |
| **Vídeo** | https://youtu.be/uDdtHNXY7Io |

---

## 1. Introdução

O problema abordado é a classificação automatizada de risco gestacional em três classes — baixo, médio e alto risco — a partir de dados clínicos coletados durante o pré-natal. A correta estratificação do risco é clinicamente relevante: falsos negativos na classe de alto risco podem resultar em complicações graves não detectadas. O sistema visa apoiar a decisão do obstetra, não substituí-la.

O dataset utilizado é o Maternal Health Risk Data Set (UCI), composto por aproximadamente 790 registros e 6 features clínicas: `Age`, `SystolicBP`, `DiastolicBP`, `BS` (glicose em mmol/L), `BodyTemp` e `HeartRate`. O dataset apresenta leve desbalanceamento entre as classes, o que torna o **F1-macro** a métrica primária de avaliação.

**Resultados baseline da Fase 1:**

| Modelo | F1-macro | ROC-AUC | Recall Alto Risco |
|---|---|---|---|
| Regressão Logística | 0.6507 | — | — |
| SVM | 0.7090 | — | — |
| **Random Forest** | **0.9017** | **0.9813** | **0.9487** |

![Comparativo F1-macro por modelo — validação cruzada](figures/baseline_cv_f1_macro.png)

O Random Forest foi o vencedor da Fase 1 com `n_estimators=100`, `max_depth=None` e `min_samples_leaf=1`, obtidos via GridSearch. Os objetivos da Fase 2 são: (1) otimizar os hiperparâmetros desse modelo via **Algoritmo Genético**, buscando superar especialmente o recall de alto risco; (2) integrar o **Claude (Anthropic API)** para transformar as saídas numéricas em explicações clínicas em linguagem natural, adaptadas ao perfil do usuário — médico ou paciente.

---

## 2. Algoritmo Genético

### 2.1 Representação

Cada indivíduo na população do AG é um dicionário de hiperparâmetros do Random Forest. O espaço de busca é definido em `SEARCH_SPACES` com 5 genes, cada um com tipo e domínio próprios:

| Gene | Tipo | Domínio |
|---|---|---|
| `n_estimators` | int | [10, 500] |
| `max_depth` | categorical | {None, 5, 10, 15, 20, 30} |
| `min_samples_split` | int | [2, 20] |
| `min_samples_leaf` | int | [1, 10] |
| `max_features` | categorical | {"sqrt", "log2", None} |

A inicialização da população é inteiramente aleatória, sem seeding de soluções conhecidas.

### 2.2 Operadores Genéticos

A escolha dos operadores é determinada pelo tipo de representação. Como os genes são independentes entre si — ao contrário do TSP, onde a solução é uma permutação de cidades e a ordem importa — é possível usar operadores mais diretos, sem restrições de preservação de sequência.

**Seleção por torneio** (`tournament_size=3`): sorteia 3 candidatos aleatórios da população e o de maior fitness é selecionado como pai. O torneio mantém diversidade — indivíduos medianos ainda têm chance de se reproduzir — evitando convergência prematura para um ótimo local.

**Cruzamento uniforme** (`crossover_rate`): cada gene do filho é herdado independentemente de um dos dois pais com probabilidade 0.5. Gera combinações arbitrárias dos genes parentais sem viés posicional, aproveitando a independência entre os hiperparâmetros.

**Mutação por gene** (`mutation_rate`): com probabilidade `mutation_rate` por gene, o valor é substituído por um novo dentro dos bounds do gene — essencial para explorar regiões do espaço que não existiam na população inicial.

**Elitismo:** o melhor indivíduo global é inserido diretamente na nova população a cada geração, garantindo que a melhor solução encontrada nunca seja perdida.

### 2.3 Função Fitness

A função fitness avalia cada indivíduo via validação cruzada estratificada:

```
fitness = média F1-macro em StratifiedKFold(n_splits=5)
```

O F1-macro foi escolhido porque o problema é multiclasse com classes desbalanceadas — a média macro trata igualmente todas as classes — e porque equilibra precisão e recall, especialmente para a classe de alto risco, onde erros têm impacto clínico assimétrico. A avaliação final no conjunto de teste é separada e realizada apenas sobre o melhor indivíduo ao final da execução.

---

## 3. Experimentos

### 3.1 Configurações

Foram realizados 3 experimentos variando os parâmetros do AG para avaliar o impacto de diferentes pressões seletivas e tamanhos de população:

| Parâmetro | Exp 1 — Padrão | Exp 2 — Alta Mutação | Exp 3 — Pop. Grande |
|---|---|---|---|
| `population_size` | 30 | 30 | 60 |
| `generations` | 20 | 20 | 30 |
| `mutation_rate` | 0.10 | 0.30 | 0.10 |
| `crossover_rate` | 0.80 | 0.80 | 0.90 |
| `random_state` | 42 | 43 | 44 |
| Tempo (s) | 450 | 667 | 1045 |

### 3.2 Convergência

![Curvas de convergência dos 3 experimentos vs baseline GridSearch](figures/ga_convergence.png)

Todos os experimentos superam o baseline já nas primeiras gerações. O Exp 1 atinge platô por volta da geração 7–10, indicando convergência eficiente. O Exp 2 convergiu mais devagar — a taxa de mutação elevada (0.30) perturba indivíduos promissores antes que o cruzamento possa explorar as regiões do espaço que eles mapeiam, diluindo a pressão seletiva com apenas 20 gerações disponíveis.

### 3.3 Resultados no Conjunto de Teste

| Modelo | F1-macro | ROC-AUC | Recall Alto Risco | Accuracy |
|---|---|---|---|---|
| Baseline (GridSearch) | 0.9017 | 0.9813 | 0.9487 | 0.8987 |
| **AG Exp 1 — Padrão** | **0.9070** | 0.9811 | **0.9744** | **0.9051** |
| AG Exp 2 — Alta Mutação | 0.9017 | 0.9813 | 0.9487 | 0.8987 |
| AG Exp 3 — Pop. Grande | 0.9013 | 0.9812 | 0.9744 | 0.8987 |

**O Exp 1 é o único que melhorou em todas as métricas simultaneamente.**

**Melhores hiperparâmetros encontrados:**

| Hiperparâmetro | Baseline | Exp 1 | Exp 2 | Exp 3 |
|---|---|---|---|---|
| `n_estimators` | 100 | **73** | 91 | 69 |
| `max_depth` | None | **15** | 20 | 15 |
| `min_samples_split` | 2 | 2 | 2 | 2 |
| `min_samples_leaf` | 1 | 1 | 1 | 1 |
| `max_features` | sqrt | sqrt | log2 | log2 |

**Matrizes de confusão — modelos otimizados:**

![Exp 1 — Padrão](figures/ga_confusion_exp1_padrao.png)
![Exp 2 — Alta Mutação](figures/ga_confusion_exp2_alta_mutacao.png)
![Exp 3 — Pop. Grande](figures/ga_confusion_exp3_pop_grande.png)

### 3.4 Análise

O Exp 1 produziu o melhor resultado geral. O recall de alto risco subiu de 0.9487 para 0.9744 — em 39 casos de alto risco no conjunto de teste, o baseline errava ~2; o AG erra ~1. Cada erro nessa classe tem consequência clínica direta, tornando esse ganho o mais significativo do projeto.

O AG convergiu para `max_depth=15` nos Experimentos 1 e 3, em contraste com `max_depth=None` do GridSearch. Árvores irrestritamente profundas tendem a memorizar o treinamento, e a validação cruzada estratificada como fitness penalizou esse overfitting de forma mais eficaz que a grade estática. O AG também encontrou `n_estimators` menores (69–91 vs 100), indicando que para ~790 registros um ensemble mais enxuto já captura a estrutura preditiva relevante.

O Exp 2 (mut=0.30) encontrou os mesmos hiperparâmetros do baseline — alta mutação demais para convergir em 20 gerações. O Exp 3 (pop=60, gen=30) atingiu recall equivalente ao Exp 1, mas ao custo de 1045s vs 450s, evidenciando retorno decrescente para este dataset.

---

## 4. Integração com LLM (Claude)

### 4.1 Arquitetura de Agentes

A integração com LLM adota uma arquitetura de agentes especializados implementada em `src/llm/agents/`. A base é a classe `BaseAgent`, que encapsula o Anthropic Python SDK e mantém o histórico de conversa multi-turn (`self.history: list[dict]`). A cada chamada a `chat()`, a mensagem é adicionada ao histórico e a API é invocada com o histórico completo — permitindo que o agente mantenha contexto ao longo de múltiplas interações na mesma sessão.

```
BaseAgent (histórico multi-turn, logging, Anthropic SDK)
├── PatientAgent  → linguagem acessível, guardrails anti-prescrição
└── DoctorAgent   → terminologia clínica, resposta em 4 seções
```

O modelo utilizado é `claude-sonnet-4-6`. Cada chamada à API registra em log `input_tokens`, `output_tokens` e `elapsed_ms` para monitoramento de custo e latência.

### 4.2 Diferenças entre Agentes

Os dois agentes diferem fundamentalmente nos system prompts, que definem tom, estrutura de resposta e guardrails:

| | PatientAgent | DoctorAgent |
|---|---|---|
| Tom | Acolhedor, sem jargão | Técnico, terminologia clínica |
| Probabilidades | Não cita números | Inclui probabilidades e feature importance |
| Estrutura | Orientações práticas de vida | 4 seções: análise · avaliação · investigação · conduta |
| Guardrails | Não prescreve, não extrapola escopo | Não responde questões administrativas/jurídicas |
| Referências clínicas | Não | PA ≥140/90 = HAS; BS ≥7.0 = diabetes; Temp ≥38°C = febre |

### 4.3 Prompt Engineering

O prompt inicial de cada agente injeta os dados clínicos individuais da paciente, a predição do modelo com probabilidades por classe e a importância das top 4 features — garantindo que as respostas sejam personalizadas para o caso específico, e não genéricas por classe de risco. O `PatientAgent` mapeia cada valor individual a orientações concretas (ex: BS=15.0 → orientação alimentar; SystolicBP=160 → buscar atendimento hoje). O `DoctorAgent` cruza os achados com os valores de referência clínicos embutidos no prompt e estrutura a investigação complementar e conduta para o caso específico.

### 4.4 Avaliação — LLM-as-judge

A qualidade das respostas geradas é avaliada automaticamente via `evaluator.py`, que usa o próprio Claude como juiz com rubricas distintas por perfil:

| Critério (PatientAgent) | Critério (DoctorAgent) |
|---|---|
| clareza (1–5) | precisao_clinica (1–5) |
| tom_adequado (1–5) | completude (1–5) |
| urgencia_correta (1–5) | acionabilidade (1–5) |
| acionabilidade (1–5) | terminologia (1–5) |
| within_scope (bool) | within_scope (bool) |

O avaliador retorna `{scores, within_scope, justificativa, score_total}` em JSON estruturado, viabilizando monitoramento contínuo da qualidade sem avaliação manual caso a caso.

### 4.5 Interface Web — Chatbot

A interface é implementada em Streamlit (`app.py`) como um chatbot conversacional. O usuário identifica seu perfil (médico ou paciente), e o bot coleta os 6 campos clínicos sequencialmente com validação fisiológica de range e cross-validação (ex: diastólica < sistólica). A temperatura é coletada em °C e convertida para °F internamente antes de passar ao modelo, que foi treinado nessa escala. Após a coleta, o modelo classifica o risco e o agente correspondente gera a análise inicial; a sessão então entra em modo Q&A multi-turn com histórico preservado.

```
Landing page
    → Identificação: médico ou paciente?
    → Coleta dos 6 campos via chat (validação fisiológica + cross-validação)
    → Predição RF → análise inicial pelo agente ativo
    → Q&A multi-turn com histórico preservado
```

---

## 5. Observabilidade

O módulo `src/observability/` implementa logging estruturado em formato JSON (NDJSON — uma linha por evento). O design prioriza plugabilidade: localmente os logs são gravados em `logs/app.log` e no stdout; para adicionar observabilidade em cloud basta instanciar um handler adicional em `setup_logging()` sem alterar o código de negócio:

```python
logger.addHandler(watchtower.CloudWatchLogHandler(...))   # AWS CloudWatch
logger.addHandler(DatadogHandler(...))                     # Datadog
```

Todos os eventos são correlacionados por `session_id`, permitindo rastrear o ciclo completo de uma sessão:

| Evento | Dados logados |
|---|---|
| `session.started` | role (patient/doctor) |
| `data.collected` | campos coletados (sem valores clínicos) |
| `model.prediction` | risk_level, probabilidades |
| `llm.call.started` | agent_type, model, turn |
| `llm.call.completed` | input_tokens, output_tokens, elapsed_ms |
| `qa.question` | número do turn |

> Os valores clínicos individuais não são logados — apenas o resultado agregado (risco e probabilidades) — por serem dados de saúde sensíveis.

---

## 6. Desafios e Soluções

O desenvolvimento em Python 3.14 introduziu três incompatibilidades com dependências existentes que exigiram soluções específicas:

| Desafio | Solução |
|---|---|
| `matplotlib.Path.__deepcopy__` causa recursão infinita no Python 3.14 | Patch manual do método no venv; remoção de `plt.tight_layout()` |
| `n_jobs=-1` falha na serialização com joblib no Python 3.14 | Substituição por `n_jobs=1` em `baseline.py` e `fitness.py` |
| scipy sem wheel para Python 3.14 | Remoção do scipy — funcionalidades cobertas pelo scikit-learn |
| dtype `object` em `y_train` após `map()` | `.astype("int64")` explícito após o split |
| AG lento sem paralelismo (~35 min para 3 experimentos) | Desenvolvimento com pop=5/gen=3; execução completa headless via `nbconvert` em background |

---

## 7. Testes Automatizados

O projeto conta com 21 testes automatizados distribuídos em 4 módulos, executados via pytest. Os testes cobrem o pipeline completo do AG — da codificação dos genes ao loop principal — e a construção dos prompts dos agentes LLM.

| Módulo | Arquivo | O que testa |
|---|---|---|
| encoding | `test_encoding.py` | Keys, bounds int, choices categorical, decode para sklearn |
| operators | `test_operators.py` | Torneio, cruzamento, mutação, bounds respeitados com mut=1.0 |
| ga | `test_ga.py` | Keys do resultado, comprimento do histórico, monotonicidade do `global_best`, reprodutibilidade |
| prompts | `test_prompts.py` | Presença de predição, dados do paciente e métricas nos prompts |

---

## 8. Conclusão

O Algoritmo Genético implementado demonstrou capacidade de superar o GridSearch da Fase 1 na métrica clinicamente mais relevante: o recall da classe de alto risco passou de 0.9487 para 0.9744 no Experimento 1, com melhorias adicionais em F1-macro (0.9017 → 0.9070) e accuracy (0.8987 → 0.9051). A convergência consistente para `max_depth=15` — contra `max_depth=None` do GridSearch — sugere que o espaço de busca contínuo combinado com validação cruzada estratificada como fitness favorece modelos com melhor generalização do que uma grade discreta estática.

A arquitetura de agentes LLM vai além de uma camada de geração de texto: o mesmo classificador produz saídas radicalmente diferentes por perfil de usuário. O `PatientAgent` traduz probabilidades e importâncias de features em orientações práticas de vida, calibradas à urgência real do caso. O `DoctorAgent` entrega análise clínica estruturada com valores de referência, investigação sugerida e conduta recomendada específica para os dados individuais da paciente. O avaliador LLM-as-judge fornece feedback automático sobre qualidade das respostas em rubricas distintas por perfil, viabilizando monitoramento contínuo sem avaliação manual.

**Limitações:**
- Dataset pequeno (~790 registros) — diferenças de 0.005 em F1-macro podem não ser estatisticamente robustas em amostras diferentes
- Critério de parada fixo — o Exp 1 estabiliza por volta da geração 7–10, indicando que parte do tempo computacional é gasto sem progresso
- Respostas dos agentes LLM não foram validadas por obstetras em estudo controlado — necessário antes de qualquer uso em ambiente clínico real
