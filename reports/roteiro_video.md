# Roteiro — Vídeo de Demonstração
## Tech Challenge FIAP Fase 2 — Otimização de Modelos de Diagnóstico

**Duração alvo:** 12–14 minutos  
**Formato:** gravação de tela + narração  
**Ferramentas sugeridas:** OBS Studio ou Loom

---

## BLOCO 1 — Contexto e Problema (2 min)
📺 _Tela: README.md no GitHub_

**Apresentação:**
- Tech Challenge Fase 2 — PosTech IA para Devs da FIAP

**Fase 1 (recap):**
- Modelos de classificação de risco gestacional
- Entrada: sinais vitais de uma gestante
- Saída: baixo, médio ou alto risco
- Vencedor: Random Forest — F1-macro 0.90, recall alto risco 0.9487

**Fase 2 — dois objetivos:**
1. Melhorar o modelo com Algoritmo Genético para otimizar hiperparâmetros
2. Integrar o Claude da Anthropic para transformar resultados numéricos em linguagem natural — adaptado para médico ou paciente

> _"Vou mostrar o código, os experimentos e o produto final funcionando."_

---

## BLOCO 2 — Estrutura do Projeto (1 min)
📺 _Tela: `tree` da pasta no terminal ou VS Code_

**Raiz:**
- `app.py` — interface web Streamlit
- `Dockerfile` — containerização

**`data/`** — dataset UCI Maternal Health Risk (790 registros, 6 sinais vitais)

**`src/`** — código separado por responsabilidade:
- `pipelines/` — carregamento e pré-processamento, split estratificado
- `models/` — pipelines de classificação (RF, LR, SVM) com scaler encadeado
- `genetic_algorithm/` — coração da Fase 2:
  - `encoding.py` — genes e bounds de busca
  - `operators.py` — seleção, cruzamento, mutação
  - `fitness.py` — avaliação com validação cruzada
  - `ga.py` — loop principal com elitismo
- `llm/agents/` — integração com Claude:
  - `base_agent.py` — histórico multi-turn
  - `patient_agent.py` / `doctor_agent.py` — agentes especializados
  - `evaluator.py` — LLM-as-judge
- `evaluation/` — métricas e gráficos
- `observability/` — logger estruturado em JSON

**`notebooks/`** — 01 baseline · 02 AG · 03 LLM

**`experiments/`** — resultados dos 3 experimentos em JSON

---

## BLOCO 3 — Algoritmo Genético: Implementação (3 min)
📺 _Tela: `encoding.py` → `operators.py` → `ga.py`_

**Representação do indivíduo** _(abrir encoding.py)_
- Cada indivíduo = dicionário com 5 genes, cada um com tipo e bounds em `SEARCH_SPACES`:
  - `n_estimators` — quantas árvores existem na floresta (10–200)
  - `max_depth` — profundidade máxima de cada árvore — controla overfitting (3–30)
  - `min_samples_split` — mínimo de amostras para dividir um nó (2–20)
  - `min_samples_leaf` — mínimo de amostras em uma folha (1–10)
  - `max_features` — quantas features considerar em cada divisão (`sqrt`, `log2` ou todas)

**Por que esses operadores?**
- A escolha dos operadores depende de como o indivíduo é representado
- Aqui os genes são **independentes** — não há restrição de ordem entre eles
- Diferente do TSP (Caixeiro Viajante), onde a solução é uma sequência de cidades e a ordem importa
- No TSP você precisa de operadores que preservem a sequência — Ordered Crossover, Swap
- Aqui podemos usar operadores mais diretos

**Seleção por torneio** _(abrir operators.py)_
- Sorteia 3 indivíduos aleatoriamente
- O com maior F1-macro vence e vira pai
- Repete para o segundo pai
- Mantém diversidade: indivíduos medianos ainda têm chance — evita convergência prematura

**Cruzamento uniforme**
- Para cada gene do filho: sorteia de qual pai herda (50/50)
- Ex: pai 1 tem `n_estimators=73`, pai 2 tem `max_depth=8` → filho herda os dois
- Gera combinações que nenhum dos pais tinha
- Funciona porque os genes são independentes

**Mutação**
- 10% de chance por gene de receber um valor novo dentro dos bounds (no experimento padrão)
- Ex: `max_depth=8` → muta para `max_depth=21`
- Fundamental para explorar regiões que não existiam na população inicial

**Loop principal** _(abrir ga.py)_
- Elitismo: o melhor indivíduo global nunca é perdido entre gerações
- Fitness: StratifiedKFold 5 folds com F1-macro
  - Problema multiclasse + classe de alto risco crítica → F1-macro é a métrica certa

---

## BLOCO 4 — Algoritmo Genético: Resultados (2 min)
📺 _Tela: `reports/figures/` aberto no Finder ou VS Code_

**3 experimentos — mencionar as configurações:**
- Exp 1 (padrão): pop=30, gen=20, mut=0.10, cross=0.80
- Exp 2 (alta mutação): pop=30, gen=20, mut=0.30, cross=0.80
- Exp 3 (pop grande): pop=60, gen=30, mut=0.10, cross=0.90

**Abrir `ga_convergence.png`**
- Gráfico da esquerda — melhor fitness global por geração:
  - Cada linha colorida = um experimento
  - Linha vermelha pontilhada = baseline GridSearch (0.9017)
  - Todos os experimentos cruzam a linha vermelha — todos superaram o GridSearch
  - Melhoria concentrada nas primeiras gerações
- Gráfico da direita — fitness médio da população:
  - A população inteira melhora, não só o melhor indivíduo

**Resultados finais — Exp 1 foi o melhor:**

| | GridSearch | AG Exp 1 |
|---|---|---|
| F1-macro | 0.9017 | **0.9070** |
| Recall alto risco | 0.9487 | **0.9744** |

- Recall de alto risco é a métrica mais importante — errar um caso grave tem consequências sérias

**Abrir `ga_confusion_exp1_padrao.png`**
- Matriz de confusão do modelo otimizado
- Mostrar que os erros em "high risk" caíram em relação ao baseline

**O que o AG encontrou que o GridSearch não encontraria:**
- `n_estimators=73` — GridSearch só testa valores da grade; AG buscou entre 10 e 200
- `max_depth=15` — árvores menores, menos overfitting que max_depth irrestrito

---

## BLOCO 5 — Arquitetura LLM (1,5 min)
📺 _Tela: `src/llm/agents/` — abrir `base_agent.py`, depois `patient_agent.py` e `doctor_agent.py` lado a lado_

**Modelo:** Claude Sonnet 4.6 (Anthropic)

**BaseAgent** _(abrir base_agent.py)_
- Mantém lista de mensagens — o histórico da conversa
- Cada chamada a `chat()` inclui o histórico completo
- O agente sabe o que foi dito antes — contexto persistente na sessão
- Loga tokens e latência de cada chamada

**Dois agentes especializados** — diferem no system prompt:

- `PatientAgent`:
  - Linguagem acessível, sem jargão
  - Guardrails: não prescreve, não faz diagnóstico definitivo, orienta atendimento presencial

- `DoctorAgent`:
  - Terminologia clínica
  - Valores de referência embutidos no prompt (PA ≥140/90, glicemia ≥7.0...)
  - Resposta estruturada em 4 seções: análise do modelo · avaliação clínica · investigação · conduta

**Evaluator** _(abrir evaluator.py)_
- LLM-as-judge: o próprio Claude avalia as respostas
- Rubricas distintas por perfil:
  - Paciente: clareza, tom, acionabilidade
  - Médico: precisão clínica, completude, terminologia
- Permite avaliar qualidade sem revisão humana caso a caso

---

## BLOCO 6 — Demo ao vivo: Paciente (3 min)
📺 _Tela: http://localhost:8501_

- Clicar **"Iniciar Avaliação"**
- Responder **"paciente"**

**Inserir dados de alto risco:**
| Campo | Valor |
|---|---|
| Idade | 48 |
| Sistólica | 160 |
| Diastólica | 100 |
| Glicemia | 15.0 |
| Temperatura | 38.5 |
| Freq. Cardíaca | 105 |

- Mostrar **validação de range** — tentar valor inválido, ver rejeição
- Mostrar **cross-validação** — diastólica menor que sistólica
- Aguardar resposta do PatientAgent:
  - Linguagem simples, sem probabilidades
  - Orientações práticas
- Digitar: _"Preciso ir ao hospital hoje ou posso esperar a consulta de amanhã?"_
  - Resposta contextualizada com os dados desta paciente — não é resposta genérica

---

## BLOCO 7 — Demo ao vivo: Médico (2 min)
📺 _Tela: App Streamlit — nova avaliação_

- Clicar **"Nova Avaliação"**
- Responder **"médico"**
- Inserir os **mesmos dados** da paciente anterior

- Aguardar resposta do DoctorAgent:
  - 4 seções clínicas estruturadas
  - Cruza PA 160/100 + idade 48 → suspeita de pré-eclâmpsia grave
  - Exames específicos sugeridos
  - Conduta com nível de urgência

- Digitar: _"Qual o risco de síndrome HELLP considerando esses achados?"_
  - O agente tem contexto completo da sessão — sabe os dados e o diagnóstico anterior

---

## BLOCO 8 — Observabilidade e Testes (1 min)
📺 _Tela: `tail -f logs/app.log` → `pytest tests/ -v`_

**Logs** _(mostrar terminal com tail)_
- JSON estruturado, um evento por linha
- `session_id` correlaciona todos os eventos da sessão:
  - `session.started` · `data.collected` · `model.prediction` · `llm.call.completed`
- Tokens e latência em cada chamada ao Claude
- Dados clínicos **não são logados** — só metadados
- Plugável: adicionar Datadog, CloudWatch ou GCP Logging = um handler em `setup_logging()`

**Testes** _(mudar para pytest)_
- 21 testes em 4 arquivos:
  - `test_encoding.py` — genes e bounds
  - `test_operators.py` — seleção, cruzamento, mutação
  - `test_ga.py` — loop principal e elitismo
  - `test_prompts.py` — construção dos prompts
- Todos passando

---

## BLOCO 9 — Encerramento (30 seg)
📺 _Tela: README no GitHub_

- Repositório no GitHub com código, notebooks executados, experimentos e relatório técnico
- Para rodar:
  1. Clonar o repositório
  2. Criar ambiente virtual e instalar dependências
  3. Configurar `ANTHROPIC_API_KEY` no `.env`
  4. `streamlit run app.py`

> _"Obrigado."_

---

## Dicas de Gravação

- **Antes de gravar:** app rodando (`streamlit run app.py`) + terminal com `tail -f logs/app.log` numa segunda janela
- **Resolução:** 1920×1080
- **Fonte VS Code:** mínimo 16pt
- **Não mostrar o `.env` nem a chave de API em nenhum momento**
