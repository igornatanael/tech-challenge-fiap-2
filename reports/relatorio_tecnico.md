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

### 4.1 Arquitetura

A integração com LLM é implementada via Anthropic Python SDK (`anthropic`). O fluxo é: cliente Anthropic instanciado com a chave de API configurada via variável de ambiente → construção do prompt via funções especializadas em `src/llm/prompts.py` → chamada à API com `model`, `system` e `messages` → retorno do texto gerado. Três funções de construção de prompt cobrem os casos de uso: `build_diagnosis_prompt` para diagnóstico individual, `build_optimization_report_prompt` para análise comparativa baseline vs. otimizado, e `build_qa_prompt` para perguntas livres do médico sobre um caso específico.

### 4.2 Prompt Engineering

**System prompt** (`SYSTEM_MEDICO`): posiciona o modelo como especialista em inteligência artificial médica assistindo obstetras. As diretrizes incluem uso de linguagem clara sem alarmismo desnecessário, preferência por terminologia reconhecível pelo médico obstetra em detrimento de jargão técnico de ML, e recomendação explícita de avaliação presencial como etapa insubstituível — reforçando que o sistema é ferramenta de apoio à decisão.

**Prompt de diagnóstico** (`build_diagnosis_prompt`): contextualiza o modelo com os dados clínicos da paciente, a predição do Random Forest, as probabilidades por classe e a importância das features. Solicita resposta estruturada em quatro seções: (1) explicação da predição em 2-3 parágrafos acessíveis ao obstetra; (2) fatores mais relevantes com significado clínico; (3) parâmetros e sinais a monitorar com maior atenção; (4) limitações do modelo e justificativa para avaliação presencial.

**Prompt Q&A** (`build_qa_prompt`): fornece o mesmo contexto do paciente (dados clínicos, predição, probabilidades por classe) e acrescenta a pergunta específica do médico. Instrui o modelo a responder de forma direta e clinicamente acionável, com indicação explícita quando a pergunta extrapola o que os dados permitem responder — prevenindo alucinações confiantes sobre informações não disponíveis.

### 4.3 Casos de Uso Demonstrados

O sistema demonstra quatro casos de uso principais. No diagnóstico individual com explicação clínica, um paciente representativo de cada classe de risco (baixo, médio e alto) é submetido ao classificador, e o LLM gera a explicação narrativa estruturada nas quatro seções descritas acima. No modo Q&A, o médico pode formular perguntas livres sobre o caso — por exemplo, sobre risco de pré-eclâmpsia ou necessidade de internação — e receber respostas contextualizadas com os dados da paciente. No relatório narrativo de otimização, o LLM analisa as métricas do baseline e do modelo otimizado pelo AG e interpreta as diferenças em termos de impacto clínico, especialmente para falsos negativos de alto risco. Por fim, um aplicativo web interativo implementado em Streamlit permite que o médico insira dados da paciente, visualize a predição do modelo e interaja com o LLM para diagnóstico e Q&A sem necessidade de linha de comando.

---

## 5. Testes Automatizados

O projeto conta com 21 testes automatizados distribuídos em 4 módulos, executados via pytest:

| Módulo de teste | Arquivo | Casos testados |
|---|---|---|
| `encoding` | `tests/test_encoding.py` | Keys do indivíduo gerado; bounds de genes int; bounds de genes float; choices de genes categorical; tamanho da população gerada; prefixo `model__` no decode para sklearn |
| `operators` | `tests/test_operators.py` | Retorno de dict com keys corretas pela seleção por torneio; seleção do melhor com tournament_size igual ao tamanho da população; ausência de cruzamento com `crossover_rate=0.0`; preservação de keys após cruzamento; preservação de keys após mutação; indivíduo inalterado com `mutation_rate=0.0`; bounds respeitados com `mutation_rate=1.0` |
| `ga` | `tests/test_ga.py` | Presença das keys esperadas no resultado; comprimento do histórico igual ao número de gerações; monotonicidade do `global_best` (nunca diminui); reprodutibilidade com mesmo `random_state` |
| `prompts` | `tests/test_prompts.py` | Presença da predição no prompt de diagnóstico; presença dos campos do paciente no prompt de diagnóstico; presença de métricas do baseline e do otimizado no prompt de relatório; presença da pergunta do médico no prompt Q&A |

---

## 6. Conclusão

O Algoritmo Genético implementado neste projeto demonstrou capacidade de superar o GridSearch da Fase 1 na métrica clinicamente mais relevante: o recall da classe de alto risco passou de 0.9487 para 0.9744 no Experimento 1, com melhora adicional no F1-macro (de 0.9017 para 0.9070) e na accuracy (de 0.8987 para 0.9051). O AG convergiu para hiperparâmetros distintos dos encontrados pelo GridSearch — em particular, profundidade máxima limitada (`max_depth=15`) contra profundidade irrestrita (`max_depth=None`) — o que sugere que o espaço de busca contínuo explorado pelo AG, combinado com a validação cruzada estratificada como função fitness, favorece modelos com melhor generalização do que a grade discreta convencional.

A integração com o Claude via Anthropic SDK adiciona uma camada de valor que vai além da otimização numérica: transforma probabilidades e importâncias de features em linguagem clínica acionável para o médico obstetra. A arquitetura de três prompts especializados — diagnóstico individual, relatório comparativo e Q&A contextualizado — cobre os principais casos de uso de um sistema de suporte à decisão clínica. O uso de um system prompt com diretrizes explícitas de comunicação médica (sem alarmismo, com indicação de avaliação presencial) é fundamental para adequar o comportamento do LLM ao contexto sensível de saúde.

As limitações do projeto são três. Primeiro, o dataset é pequeno (~790 registros), o que limita a robustez estatística das comparações entre configurações do AG — diferenças de 0.005 no F1-macro podem não ser reproduzíveis em amostras diferentes. Segundo, o critério de parada do AG é fixo (número de gerações), sem critério de convergência adaptativo; os experimentos mostram que o best fitness se estabiliza por volta da geração 7-10 no Experimento 1, indicando que parte do tempo computacional é gasto sem progresso. Terceiro, as explicações geradas pelo LLM não passam por validação clínica formal: são plausíveis e bem estruturadas, mas não foram avaliadas por obstetras em estudo controlado, o que seria necessário antes de qualquer uso em ambiente clínico real.
