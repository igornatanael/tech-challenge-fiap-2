# Relatório Técnico — Tech Challenge FIAP Fase 2

## Informações

- **Curso:** PosTech IA para Devs — FIAP
- **Fase:** 2 — Evolução da IA: GenAI, Cloud ML e LLMs
- **Projeto:** Projeto 1 — Otimização de Modelos de Diagnóstico
- **Dataset:** Maternal Health Risk (UCI) — ~790 registros, 6 features clínicas

---

## 1. Introdução

O problema abordado é a classificação automatizada de risco gestacional em três classes — baixo, médio e alto risco — a partir de dados clínicos coletados durante o pré-natal. A correta estratificação do risco é clinicamente relevante: falsos negativos na classe de alto risco podem resultar em complicações graves não detectadas, enquanto falsos positivos em excesso sobrecarregam serviços hospitalares especializados. O sistema visa apoiar a decisão do obstetra, não substituí-la.

O dataset utilizado é o Maternal Health Risk Data Set (UCI), composto por aproximadamente 790 registros e 6 features clínicas: Age (idade em anos), SystolicBP (pressão arterial sistólica, mmHg), DiastolicBP (pressão arterial diastólica, mmHg), BS (glicose sanguínea, mmol/L), BodyTemp (temperatura corporal, °F) e HeartRate (frequência cardíaca, bpm). O alvo é uma variável categórica com três classes de risco gestacional. O dataset apresenta leve desbalanceamento entre as classes, o que torna o F1-macro a métrica primária de avaliação.

Na Fase 1, três classificadores foram treinados e avaliados via GridSearch com validação cruzada estratificada: Regressão Logística (F1-macro 0.6507), SVM (F1-macro 0.7090) e Random Forest (F1-macro 0.9017). O Random Forest, com hiperparâmetros `n_estimators=100`, `max_depth=None` e `min_samples_leaf=1`, foi o modelo vencedor, atingindo também ROC-AUC de 0.9813 e recall de alto risco de 0.9487.

O objetivo da Fase 2 é duplo: (1) otimizar os hiperparâmetros do Random Forest via Algoritmo Genético (AG), buscando superar os resultados do GridSearch da Fase 1 — em particular o recall da classe de alto risco; (2) integrar um modelo de linguagem de grande escala (LLM), especificamente o Claude via Anthropic API, para transformar as saídas numéricas do classificador em explicações clínicas em linguagem natural direcionadas ao médico obstetra.

---

## 2. Algoritmo Genético

### 2.1 Representação (Codificação)

Cada indivíduo na população do AG é um dicionário de hiperparâmetros do Random Forest. O espaço de busca é definido em `src/genetic_algorithm/encoding.py` no dicionário `SEARCH_SPACES`, com 5 genes para o modelo `random_forest`:

| Gene | Tipo | Domínio |
|---|---|---|
| `n_estimators` | int | [10, 500] |
| `max_depth` | categorical | {None, 5, 10, 15, 20, 30} |
| `min_samples_split` | int | [2, 20] |
| `min_samples_leaf` | int | [1, 10] |
| `max_features` | categorical | {"sqrt", "log2", None} |

Genes do tipo `int` são amostrados por `random.randint(low, high)`. Genes do tipo `categorical` são amostrados por `random.choice(choices)`. A inicialização da população é inteiramente aleatória, sem seeding de soluções conhecidas.

### 2.2 Operadores Genéticos

**Seleção por torneio** (`tournament_selection`, `tournament_size=3`): amostra-se aleatoriamente 3 candidatos da população corrente e o de maior fitness é selecionado como pai. O processo é repetido independentemente para selecionar o segundo pai. A seleção por torneio foi escolhida por ser robusta a diferenças de escala no fitness e por não requerer normalização da distribuição de aptidão, ao contrário da seleção por roleta.

**Cruzamento uniforme** (`uniform_crossover`): controlado pelo parâmetro `crossover_rate`. Se um valor aleatório uniforme superar `crossover_rate`, os pais são retornados sem cruzamento. Caso contrário, cada gene do filho é herdado independentemente de um dos dois pais com probabilidade 0.5. O método produz dois filhos por par de pais, permitindo que a nova geração herde combinações arbitrárias dos genes parentais, sem o viés posicional do cruzamento de um ponto.

**Mutação por gene** (`mutate`): aplica-se com probabilidade `mutation_rate` a cada gene individualmente. Para genes do tipo `int`, o novo valor é amostrado uniformemente dentro dos bounds. Para genes do tipo `float`, aplica-se perturbação gaussiana com desvio-padrão igual a 20% do intervalo do gene, seguida de clipping nos bounds. Para genes do tipo `categorical`, uma nova opção é escolhida aleatoriamente entre as `choices` definidas.

### 2.3 Função Fitness

A função fitness avalia cada indivíduo via validação cruzada estratificada com `StratifiedKFold(n_splits=5)`, utilizando `scoring='f1_macro'` e `n_jobs=1`. O F1-macro foi escolhido como métrica primária por duas razões: (1) o problema é multiclasse com três classes com distribuições distintas, e a média macro trata igualmente todas as classes independentemente de sua frequência; (2) é necessário equilibrar precisão e recall especialmente para a classe de alto risco, onde erros têm impacto clínico assimétrico.

O valor retornado pela função fitness é a média do F1-macro nos 5 folds de validação cruzada. A avaliação final no conjunto de teste é separada e realizada apenas sobre o melhor indivíduo encontrado ao final da execução do AG.

### 2.4 Loop Principal

O AG executa um número fixo de gerações como critério de parada. A cada geração: (1) toda a população é avaliada pela função fitness via validação cruzada; (2) o melhor indivíduo da geração é comparado com o melhor global e, se superior, atualiza o global; (3) uma nova população é construída via seleção por torneio, cruzamento uniforme e mutação por gene. Com elitismo habilitado (`elitism=True`), o melhor indivíduo global é inserido diretamente na nova população antes da geração de filhos, garantindo que a solução atual nunca seja perdida. A última geração não gera descendentes — o algoritmo retorna o melhor indivíduo e o histórico de convergência.

---

## 3. Experimentos

### 3.1 Configurações

| Parâmetro | Experimento 1 | Experimento 2 | Experimento 3 |
|---|---|---|---|
| Descrição | Configuração padrão | Alta mutação | População grande |
| `population_size` | 30 | 30 | 60 |
| `generations` | 20 | 20 | 30 |
| `mutation_rate` | 0.1 | 0.3 | 0.1 |
| `crossover_rate` | 0.8 | 0.8 | 0.9 |
| `tournament_size` | 3 | 3 | 3 |
| `elitism` | True | True | True |
| `random_state` | 42 | 43 | 44 |

### 3.2 Resultados

**Tabela comparativa — métricas no conjunto de teste:**

| Configuração | F1-macro | ROC-AUC (macro OVR) | Recall Alto Risco | Accuracy | Fitness CV (melhor) | Tempo (s) |
|---|---|---|---|---|---|---|
| Baseline RF (GridSearch) | 0.9017 | 0.9813 | 0.9487 | 0.8987 | — | — |
| Experimento 1 — Padrão | 0.9070 | 0.9811 | 0.9744 | 0.9051 | 0.7963 | 450.2 |
| Experimento 2 — Alta Mutação | 0.9017 | 0.9813 | 0.9487 | 0.8987 | 0.7921 | 666.5 |
| Experimento 3 — Pop. Grande | 0.9013 | 0.9812 | 0.9744 | 0.8987 | 0.7993 | 1045.3 |

**Melhores hiperparâmetros encontrados:**

| Hiperparâmetro | Baseline (GridSearch) | Experimento 1 | Experimento 2 | Experimento 3 |
|---|---|---|---|---|
| `n_estimators` | 100 | 73 | 91 | 69 |
| `max_depth` | None | 15 | 20 | 15 |
| `min_samples_split` | — | 2 | 2 | 2 |
| `min_samples_leaf` | 1 | 1 | 1 | 1 |
| `max_features` | — | sqrt | log2 | log2 |

*Nota: o GridSearch da Fase 1 não buscou `min_samples_split` nem `max_features` explicitamente; esses parâmetros ficaram nos valores default do sklearn.*

### 3.3 Análise

O Experimento 1 produziu o melhor resultado geral: F1-macro de 0.9070 contra 0.9017 do baseline, accuracy de 0.9051 contra 0.8987, e recall da classe de alto risco elevado de 0.9487 para 0.9744. A melhoria no recall de alto risco é a mais relevante clinicamente — o modelo passou a identificar corretamente 97,44% dos casos de alto risco no conjunto de teste, reduzindo a taxa de falsos negativos nessa classe crítica. Isso representa um ganho concreto sobre o GridSearch com custo computacional aceitável (450 segundos).

O AG convergiu para árvores com `max_depth=15` nos Experimentos 1 e 3, em contraste com `max_depth=None` encontrado pelo GridSearch. Árvores irrestritamente profundas tendem a memorizar o conjunto de treinamento, e o fato de o AG ter preferido consistentemente profundidade limitada sugere que a validação cruzada estratificada de 5 folds penalizou o overfitting de maneira mais eficaz do que a grade estática usada na Fase 1. Adicionalmente, o AG encontrou `n_estimators` menores (69–91) contra os 100 do GridSearch, indicando que para esse dataset um ensemble mais enxuto já captura a estrutura preditiva relevante sem custo adicional de treinamento.

O Experimento 2, com taxa de mutação elevada (0.3 contra 0.1), não melhorou sobre o baseline: obteve F1-macro de 0.9017 e recall de alto risco de 0.9487 — idênticos ao GridSearch da Fase 1. A alta taxa de mutação introduz diversidade excessiva a cada geração, perturbando indivíduos promissores antes que o operador de cruzamento possa explotar as regiões do espaço de busca que eles mapeiam. Com apenas 20 gerações disponíveis, a convergência foi insuficiente: o fitness CV máximo atingido foi 0.7921 contra 0.7963 do Experimento 1. A convergência mais lenta do Experimento 2 é visível no histórico: o fitness médio por geração manteve-se consistentemente abaixo do obtido no Experimento 1, indicando que a pressão seletiva foi diluída pela perturbação excessiva.

O Experimento 3, com população de 60 indivíduos e 30 gerações, convergiu para uma solução com recall de alto risco equivalente ao Experimento 1 (0.9744), mas com F1-macro ligeiramente inferior (0.9013) e precisão macro menor (0.8976). O fitness CV final (0.7993) foi o maior entre os três experimentos, indicando melhor generalização na validação cruzada, ainda que a diferença no conjunto de teste seja marginal. O custo foi significativo: 1045.3 segundos, mais que o dobro do Experimento 1. Isso sugere que, para este dataset de ~790 registros, o aumento de população além de 30 indivíduos oferece retorno decrescente. O Experimento 1 apresenta a melhor relação entre desempenho e custo computacional.

---

## 4. Integração com LLM (Claude)

### 4.1 Arquitetura de Agentes

A integração com LLM adota uma arquitetura de agentes especializados implementada em `src/llm/agents/`. A base é a classe `BaseAgent`, que encapsula o Anthropic Python SDK (`anthropic==0.40.0`) e mantém o histórico de conversa multi-turn (`self.history: list[dict]`). A cada chamada a `chat()`, a mensagem do usuário é adicionada ao histórico, a API é invocada com o histórico completo e o `system` prompt do agente, e a resposta do Claude é adicionada de volta — permitindo que o agente mantenha contexto ao longo de múltiplas interações na mesma sessão.

Dois agentes concretos herdam de `BaseAgent`:

**`PatientAgent`** — projetado para interagir com a paciente gestante. System prompt com tom acolhedor, linguagem acessível sem jargão médico, guardrail explícito contra prescrição de medicamentos e contra perguntas fora do escopo de saúde gestacional. Ao receber uma pergunta fora do escopo, retorna mensagem padronizada sem tentar responder.

**`DoctorAgent`** — projetado para interagir com o médico obstetra. System prompt com tom técnico e terminologia clínica, valores de referência clínicos embutidos (ex: PA ≥140/90 = HAS, BS ≥7.0 mmol/L = diabetes), guardrail contra perguntas administrativas ou jurídicas. Estrutura as respostas iniciais em quatro seções: análise do modelo, avaliação clínica, investigação sugerida e conduta recomendada.

O modelo utilizado é `claude-sonnet-4-6`, configurável via variável de ambiente `ANTHROPIC_MODEL`.

### 4.2 Prompt Engineering

Cada agente dispõe de uma função de construção do prompt inicial (`_build_*_initial_prompt`) que injeta os dados clínicos individuais da paciente, a predição do modelo, as probabilidades por classe e a importância das features — permitindo respostas personalizadas para cada caso, e não genéricas por classe de risco.

**PatientAgent:** o prompt instrui o Claude a explicar o risco sem citar probabilidades numéricas, mapear os valores individuais (ex: BS=15.0 mmol/L → orientação sobre alimentação; SystolicBP=160 mmHg → orientação de buscar atendimento hoje) e calibrar a urgência da recomendação ao nível de risco real (emergência / urgência / rotina).

**DoctorAgent:** o prompt fornece a importância das top 4 features com os valores da paciente, solicita análise por seção com interpretação dos achados individuais em relação aos valores de referência clínicos, e orienta investigação complementar e conduta específicas para o caso.

### 4.3 Avaliação da Qualidade (LLM-as-judge)

A qualidade das respostas geradas pelos agentes é avaliada automaticamente via `src/llm/agents/evaluator.py`, utilizando o próprio Claude como juiz. O avaliador recebe a resposta gerada, o tipo de agente, os dados do caso e o nível de risco, e retorna scores em rubric distinto por agente:

| Critério (PatientAgent) | Critério (DoctorAgent) |
|---|---|
| clareza (1–5) | precisao_clinica (1–5) |
| tom_adequado (1–5) | completude (1–5) |
| urgencia_correta (1–5) | acionabilidade (1–5) |
| acionabilidade (1–5) | terminologia (1–5) |
| within_scope (bool) | within_scope (bool) |

O avaliador faz uma única chamada à API, solicita resposta em JSON estruturado e parseia o resultado, retornando `{scores, within_scope, justificativa, score_total}`. O `score_total` é a média aritmética dos critérios numéricos.

### 4.4 Interface Web — Chatbot

A interface de usuário é implementada em Streamlit (`app.py`) como um chatbot conversacional, substituindo o formulário estático original. O fluxo é:

1. **Landing page** com botão "Iniciar Avaliação"
2. **Identificação do perfil**: o bot pergunta se o usuário é médico ou paciente, instanciando o agente correspondente
3. **Coleta de dados via chat**: os 6 campos clínicos são coletados sequencialmente com validação de range fisiológico e cross-validation (ex: pressão diastólica < sistólica). A temperatura é coletada em °C e convertida para °F internamente antes de passar ao modelo, que foi treinado nessa escala
4. **Diagnóstico**: o modelo RF otimizado classifica o risco; o agente ativo gera a análise inicial personalizada
5. **Q&A multi-turn**: sessão aberta de perguntas com histórico de conversa preservado

Palavras-chave como "recomeçar" ou "nova avaliação" reiniciam a sessão a qualquer momento.

---

## 5. Observabilidade

O módulo `src/observability/` implementa logging estruturado em formato JSON (NDJSON — uma linha por evento) via o módulo `logging` nativo do Python. O design prioriza plugabilidade: localmente os logs são gravados em `logs/app.log` e no stdout; para adicionar observabilidade cloud basta instanciar um handler adicional em `setup_logging()` sem alterar o código de negócio:

```python
# AWS CloudWatch
logger.addHandler(watchtower.CloudWatchLogHandler(log_group="/risk-gestacional"))

# Datadog
logger.addHandler(DatadogHandler(api_key=..., service="risk-gestacional"))

# OpenTelemetry (vendor-neutral)
# configurar LoggerProvider com o exporter do backend desejado
```

Eventos logados por sessão, todos com `session_id` para correlação:

| Evento | Dados |
|---|---|
| `session.started` | role (patient/doctor) |
| `data.collected` | lista de campos coletados (sem valores clínicos) |
| `model.prediction` | risk_level, probabilidades |
| `llm.call.started` | agent_type, model, turn |
| `llm.call.completed` | agent_type, input_tokens, output_tokens, elapsed_ms |
| `qa.question` | turn da pergunta |

Os valores clínicos individuais não são logados — apenas o resultado agregado (risco e probabilidades) — por serem dados de saúde sensíveis.

---

## 6. Desafios e Soluções

### 6.1 Incompatibilidade do matplotlib com Python 3.14

O método `Path.__deepcopy__` do matplotlib chama `copy.deepcopy(super(), memo)`, que em Python 3.14 causa recursão infinita devido a uma mudança no comportamento do proxy `super()` no módulo `copy`. O erro se manifestava em qualquer renderização de figura, impedindo a execução dos notebooks.

**Solução:** patch direto no arquivo do matplotlib instalado no ambiente virtual, substituindo a implementação do método por uma cópia manual via `__dict__`, que não aciona o mecanismo de deepcopy recursivo. Adicionalmente, todas as chamadas a `plt.tight_layout()` foram removidas dos notebooks — eram desnecessárias pois `bbox_inches="tight"` no `savefig` já cumpre a mesma função — e `plt.show()` foi substituído por `plt.close("all")` para liberar memória em execução headless.

### 6.2 Serialização com joblib no Python 3.14

A paralelização via `n_jobs=-1` no scikit-learn (usado no GridSearch e na validação cruzada) causava falhas de serialização com joblib no Python 3.14. O erro não ocorria no Python 3.11, evidenciando uma incompatibilidade com a versão em uso.

**Solução:** substituição de todos os `n_jobs=-1` por `n_jobs=1` em `src/models/baseline.py` e `src/genetic_algorithm/fitness.py`. O impacto no tempo de execução é relevante para os experimentos do AG (cada avaliação de fitness roda sequencialmente), mas é aceitável dado o tamanho do dataset (~790 registros) e a ausência de requisito de tempo real.

### 6.3 Ausência de wheel do scipy para Python 3.14

O scipy não disponibilizava wheel compilado para Python 3.14 no momento do desenvolvimento, e a compilação a partir do código-fonte falhou por incompatibilidades com o toolchain disponível.

**Solução:** remoção do scipy do `requirements.txt`. As funcionalidades utilizadas no projeto (validação cruzada, métricas, pipelines) são cobertas pelo scikit-learn sem dependência direta do scipy.

### 6.4 Dtype do alvo após mapeamento categórico

O pipeline de pré-processamento converte a coluna de risco de strings (`"low risk"`, `"mid risk"`, `"high risk"`) para inteiros via `map()`. Após o split treino/teste, o dtype resultante era `object` em vez de `int64`, causando `ValueError: Got 'unknown' target type` no scikit-learn ao tentar treinar classificadores.

**Solução:** adição de `.astype("int64")` explícito em `y_train` e `y_test` imediatamente após o split em `src/pipelines/preprocessing.py`.

### 6.5 Custo computacional do AG sem paralelismo

Com `n_jobs=1` e 3 experimentos (600 + 600 + 1800 = 3000 avaliações de fitness), o tempo total de execução dos experimentos foi de aproximadamente 35 minutos. Isso inviabilizou iteração rápida durante o desenvolvimento.

**Solução:** para o ciclo de desenvolvimento, os experimentos foram rodados com populações e gerações reduzidas (pop=5, gen=3) para validar o pipeline completo. A execução completa foi realizada via `nbconvert` headless em background, permitindo trabalho paralelo. Os resultados foram salvos em JSON após cada experimento para evitar re-execução.

---

## 7. Testes Automatizados

O projeto conta com 21 testes automatizados distribuídos em 4 módulos, executados via pytest:

| Módulo de teste | Arquivo | Casos testados |
|---|---|---|
| `encoding` | `tests/test_encoding.py` | Keys do indivíduo gerado; bounds de genes int; bounds de genes float; choices de genes categorical; tamanho da população gerada; prefixo `model__` no decode para sklearn |
| `operators` | `tests/test_operators.py` | Retorno de dict com keys corretas pela seleção por torneio; seleção do melhor com tournament_size igual ao tamanho da população; ausência de cruzamento com `crossover_rate=0.0`; preservação de keys após cruzamento; preservação de keys após mutação; indivíduo inalterado com `mutation_rate=0.0`; bounds respeitados com `mutation_rate=1.0` |
| `ga` | `tests/test_ga.py` | Presença das keys esperadas no resultado; comprimento do histórico igual ao número de gerações; monotonicidade do `global_best` (nunca diminui); reprodutibilidade com mesmo `random_state` |
| `prompts` | `tests/test_prompts.py` | Presença da predição no prompt de diagnóstico; presença dos campos do paciente no prompt de diagnóstico; presença de métricas do baseline e do otimizado no prompt de relatório; presença da pergunta do médico no prompt Q&A |

---

## 7. Conclusão

O Algoritmo Genético implementado neste projeto demonstrou capacidade de superar o GridSearch da Fase 1 na métrica clinicamente mais relevante: o recall da classe de alto risco passou de 0.9487 para 0.9744 no Experimento 1, com melhora adicional no F1-macro (de 0.9017 para 0.9070) e na accuracy (de 0.8987 para 0.9051). O AG convergiu para hiperparâmetros distintos dos encontrados pelo GridSearch — em particular, profundidade máxima limitada (`max_depth=15`) contra profundidade irrestrita (`max_depth=None`) — o que sugere que o espaço de busca contínuo explorado pelo AG, combinado com a validação cruzada estratificada como função fitness, favorece modelos com melhor generalização do que a grade discreta convencional.

A integração com o Claude via Anthropic SDK vai além de uma camada de geração de texto: a arquitetura de agentes especializados (`PatientAgent` e `DoctorAgent`) permite que o mesmo modelo de classificação produza saídas radicalmente diferentes dependendo do perfil do usuário. O `PatientAgent` traduz probabilidades e importâncias de features em orientações práticas de vida, calibradas à urgência real do caso. O `DoctorAgent` entrega análise clínica estruturada com valores de referência, investigação sugerida e conduta recomendada — específica para os dados individuais da paciente, não genérica por classe de risco. O avaliador LLM-as-judge (`evaluator.py`) fornece feedback automático sobre a qualidade das respostas em rubrics distintos por perfil, viabilizando monitoramento contínuo da qualidade sem avaliação manual. A interface chatbot coleta dados conversacionalmente com validação fisiológica, oferecendo uma experiência mais natural que formulários estáticos tanto para pacientes quanto para profissionais de saúde.

As limitações do projeto são três. Primeiro, o dataset é pequeno (~790 registros), o que limita a robustez estatística das comparações entre configurações do AG — diferenças de 0.005 no F1-macro podem não ser reproduzíveis em amostras diferentes. Segundo, o critério de parada do AG é fixo (número de gerações), sem critério de convergência adaptativo; os experimentos mostram que o best fitness se estabiliza por volta da geração 7-10 no Experimento 1, indicando que parte do tempo computacional é gasto sem progresso. Terceiro, as explicações geradas pelos agentes LLM não passam por validação clínica formal: são plausíveis e bem estruturadas, mas não foram avaliadas por obstetras em estudo controlado, o que seria necessário antes de qualquer uso em ambiente clínico real.
