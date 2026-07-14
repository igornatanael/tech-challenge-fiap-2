# Roteiro — Vídeo de Demonstração
## Tech Challenge FIAP Fase 2 — Otimização de Modelos de Diagnóstico

**Duração alvo:** 12–14 minutos  
**Formato:** gravação de tela + narração  
**Ferramentas sugeridas:** OBS Studio ou Loom

---

## BLOCO 1 — Contexto e Problema (2 min)

**O que mostrar na tela:** README.md aberto no GitHub

**Fala:**

> "Olá. Este é o Tech Challenge da Fase 2 do PosTech IA para Devs da FIAP.
>
> Na Fase 1, construímos modelos de classificação de risco gestacional — dado um conjunto de sinais vitais de uma gestante, o modelo prevê se ela está em baixo, médio ou alto risco. O vencedor foi um Random Forest com F1-macro de 0.90 e recall de alto risco de 0.9487.
>
> Na Fase 2, dois objetivos: primeiro, melhorar esse modelo usando Algoritmo Genético para otimizar os hiperparâmetros. Segundo, integrar o Claude da Anthropic para transformar os resultados numéricos em explicações em linguagem natural — adaptadas para médico ou para a própria paciente.
>
> Vou mostrar o código, os experimentos e o produto final funcionando."

---

## BLOCO 2 — Estrutura do Projeto (1 min)

**O que mostrar na tela:** VS Code ou terminal com `tree` da pasta do projeto

**Fala:**

> "A estrutura está organizada assim:"
>
> "Na raiz temos o `app.py` — a interface web em Streamlit — e o `Dockerfile` para containerização."
>
> "Em `data/` fica o dataset do UCI: Maternal Health Risk, com 790 registros e 6 sinais vitais."
>
> "Em `src/` está todo o código de negócio, separado por responsabilidade:"
>
> "— `pipelines/` cuida do carregamento e pré-processamento dos dados, incluindo o split treino/teste estratificado."
>
> "— `models/` define os pipelines de classificação — Random Forest, Regressão Logística e SVM — com o scaler já encadeado."
>
> "— `genetic_algorithm/` é o coração da Fase 2: `encoding.py` define os genes e os bounds de busca; `operators.py` implementa seleção por torneio, cruzamento uniforme e mutação; `fitness.py` avalia cada indivíduo com validação cruzada; e `ga.py` é o loop principal com elitismo."
>
> "— `llm/` contém a integração com o Claude: `client.py` é o singleton que gerencia a conexão com a API da Anthropic; dentro de `agents/` temos `base_agent.py` com o histórico multi-turn, `patient_agent.py` e `doctor_agent.py` com os system prompts e guardrails especializados, e `evaluator.py` que funciona como LLM-as-judge."
>
> "— `evaluation/` tem as métricas e os gráficos de avaliação dos modelos."
>
> "— `observability/` tem o logger estruturado em JSON que registra cada evento da sessão."
>
> "Os `notebooks/` documentam cada etapa em ordem: 01 é o baseline, 02 é o algoritmo genético, 03 é a integração com LLM."
>
> "Em `experiments/` ficam os resultados dos 3 experimentos do AG salvos como JSON."
>
> "E em `reports/figures/` estão os gráficos gerados — curvas de convergência, matrizes de confusão."

---

## BLOCO 3 — Algoritmo Genético: Implementação (3 min)

**O que mostrar na tela:** `src/genetic_algorithm/encoding.py` → `operators.py` → `ga.py`

**Fala:**

> "O Algoritmo Genético otimiza os hiperparâmetros do Random Forest. Cada indivíduo na população é um dicionário com 5 genes — n_estimators, max_depth, min_samples_split, min_samples_leaf e max_features — com bounds e tipos definidos aqui no SEARCH_SPACES."
>
> "Antes de falar dos operadores, é importante entender por que escolhemos esses operadores específicos. A escolha depende diretamente de como o indivíduo é representado. No nosso caso, os genes são independentes entre si — não existe restrição de ordem entre eles. Isso é diferente do exemplo clássico de Algoritmos Genéticos, o Problema do Caixeiro Viajante, onde a solução é uma sequência de cidades e a ordem importa. Lá, você precisa de operadores que preservem essa ordem — como o Ordered Crossover e a mutação Swap. No nosso problema, como cada hiperparâmetro é independente dos outros, podemos usar operadores mais simples e diretos."
>
> *(abrir operators.py)*
>
> "O primeiro operador é a **seleção por torneio**. Em vez de ordenar toda a população por fitness, sorteamos 3 indivíduos aleatoriamente e o com maior F1-macro vence — e vira pai. Repetimos isso para escolher o segundo pai. A vantagem do torneio é que ele mantém diversidade na população: indivíduos medianos ainda têm chance de ser selecionados, o que evita que o AG convirja rápido demais para um ótimo local. A seleção por torneio funciona bem para qualquer tipo de problema — ela é a parte mais universal dos três operadores."
>
> "O segundo é o **cruzamento uniforme**. Para cada gene do filho, sorteamos uma moeda — cara herda do pai 1, coroa herda do pai 2. Por exemplo: o pai 1 tem n_estimators 73 e max_depth 15; o pai 2 tem n_estimators 120 e max_depth 8. O filho pode herdar n_estimators 73 do pai 1 e max_depth 8 do pai 2 — uma combinação que nenhum dos dois tinha. Cada gene é decidido de forma independente, com 50% de chance de vir de cada pai. Essa independência é exatamente o que nos permite usar o cruzamento uniforme — ela seria problemática no TSP, onde trocar genes de forma independente quebraria a sequência de cidades."
>
> "O terceiro é a **mutação**. Com probabilidade de 20% por gene, o valor é substituído por um novo valor aleatório dentro dos bounds daquele gene. Por exemplo, um max_depth que era 8 pode mutar para 21. A mutação é fundamental para que o AG explore regiões do espaço que não existiam na população inicial — sem ela, o algoritmo só recombinaria o que já tem, nunca descobrindo valores inéditos."
>
> *(abrir ga.py, mostrar o loop)*
>
> "O loop principal usa elitismo — o melhor indivíduo global nunca é perdido entre gerações. E a função fitness avalia cada indivíduo com StratifiedKFold de 5 folds, usando F1-macro como critério — importante porque o problema é multiclasse e a classe de alto risco é crítica."

---

## BLOCO 4 — Algoritmo Genético: Resultados (2 min)

**O que mostrar na tela:** Notebook 02 executado — curvas de convergência e tabela comparativa

**Fala:**

> "Rodamos 3 experimentos variando as configurações do AG. O Experimento 1 usou a configuração padrão — população de 30 indivíduos, 20 gerações, taxa de mutação de 20%. O Experimento 2 aumentou a taxa de mutação para forçar mais exploração. O Experimento 3 dobrou o tamanho da população para ver se mais diversidade inicial ajudava."
>
> "Aqui estão as curvas de convergência — cada linha mostra como o melhor fitness evoluiu ao longo das gerações em cada experimento. A linha vermelha pontilhada é o baseline do GridSearch. Dá pra ver que todos os experimentos superaram o baseline, e que a melhoria acontece principalmente nas primeiras gerações — sinal de que o espaço de busca não é tão complexo a ponto de precisar de muitas gerações."
>
> *(rolar até a tabela comparativa)*
>
> "O Experimento 1 foi o melhor. F1-macro subiu de 0.9017 para 0.9070, e o recall de alto risco — que é a métrica mais importante clinicamente, porque errar um caso grave tem consequências sérias — subiu de 0.9487 para 0.9744."
>
> "O que o AG encontrou de diferente do GridSearch? Árvores menores: max_depth 15 em vez de sem limite, n_estimators 73 em vez de 100. O GridSearch nunca testaria 73 árvores — ele só testa os valores que você listou explicitamente na grade. O AG buscou num espaço contínuo entre 10 e 200 e convergiu para 73. A validação cruzada de 5 folds dentro da função fitness penalizou overfitting de forma mais eficaz do que a avaliação estática do GridSearch. Os resultados completos de cada experimento ficaram salvos em JSON aqui em `experiments/`."

---

## BLOCO 5 — Arquitetura LLM (1,5 min)

**O que mostrar na tela:** `src/llm/agents/` — mostrar os arquivos, abrir `base_agent.py` e depois `patient_agent.py` e `doctor_agent.py` lado a lado

**Fala:**

> "Para a integração com LLM, usamos o Claude da Anthropic — especificamente o claude-sonnet-4-6. A arquitetura é baseada em dois agentes especializados que herdam de uma classe base comum."
>
> *(abrir base_agent.py)*
>
> "O `BaseAgent` é a fundação. Ele mantém uma lista de mensagens — o histórico da conversa. Toda vez que o método `chat()` é chamado, ele inclui o histórico completo na requisição para o Claude. Isso é o que permite que o agente mantenha contexto ao longo de uma sessão — ele sabe o que foi perguntado e respondido antes. Também registra em log cada chamada com tokens consumidos e latência."
>
> *(abrir patient_agent.py e doctor_agent.py lado a lado)*
>
> "Os dois agentes especializados diferem principalmente nos system prompts — a instrução que define o comportamento do modelo. O `PatientAgent` usa linguagem acessível, sem jargão médico, e tem guardrails explícitos no prompt: não prescreve medicamentos, não faz diagnóstico definitivo, sempre orienta a buscar atendimento presencial. O `DoctorAgent` usa terminologia clínica, tem valores de referência embutidos no system prompt — como PA acima de 140/90 indica hipertensão, glicemia acima de 7.0 indica diabetes — e estrutura a resposta em 4 seções: análise do modelo, avaliação clínica, investigação sugerida e conduta recomendada."
>
> "Além dos dois agentes, temos o `evaluator.py` — um LLM-as-judge. Ele usa o próprio Claude para avaliar a qualidade das respostas geradas, com rubricas diferentes para cada perfil. Para o PatientAgent avalia clareza, tom adequado e acionabilidade. Para o DoctorAgent avalia precisão clínica, completude e terminologia. Isso permite medir objetivamente a qualidade das respostas sem depender de avaliação humana para cada caso."

---

## BLOCO 6 — Demo ao vivo: Paciente (3 min)

**O que mostrar na tela:** App Streamlit em http://localhost:8501

**Fala:**

> "Agora a demo ao vivo. Essa é a interface — um chatbot que coleta os dados da paciente conversacionalmente."
>
> *(clicar em "Iniciar Avaliação")*
>
> "O bot pergunta primeiro se é médico ou paciente. Vou simular uma paciente."
>
> *(digitar "paciente")*
>
> "Agora ele coleta os dados um a um. Vou inserir o perfil de uma paciente de alto risco para mostrar a validação e o diagnóstico."
>
> *(inserir: Idade 48 → Sistólica 160 → Diastólica 100 → Glicemia 15.0 → Temperatura 38.5 → FC 105)*
>
> "Perceba que se eu tentar colocar um valor fora dos limites fisiológicos ele rejeita e explica o motivo. E aqui a cross-validação — diastólica tem que ser menor que a sistólica."
>
> *(aguardar diagnóstico do PatientAgent aparecer)*
>
> "A resposta do PatientAgent é em linguagem simples — não cita probabilidades, traduz os achados em orientações práticas. E no fim pergunta se tem dúvidas. Vou fazer uma pergunta."
>
> *(digitar: "Preciso ir ao hospital hoje ou posso esperar a consulta de amanhã?")*
>
> "A resposta é contextualizada com os dados desta paciente específica — não é uma resposta genérica de alto risco."

---

## BLOCO 7 — Demo ao vivo: Médico (2 min)

**O que mostrar na tela:** App Streamlit — nova avaliação

**Fala:**

> "Agora vou mostrar o fluxo para médico com o mesmo perfil de paciente."
>
> *(clicar "Nova Avaliação", digitar "médico", inserir mesmos dados)*
>
> *(aguardar diagnóstico do DoctorAgent)*
>
> "Compare com a resposta anterior. O DoctorAgent estrutura em seções clínicas: análise do modelo com feature importance, avaliação dos achados — aqui ele cruza a PA 160/100 com a idade de 48 anos e aponta suspeita de pré-eclâmpsia grave. Investigação sugerida com exames específicos para o caso. E conduta recomendada com nível de urgência."
>
> *(digitar: "Qual o risco de síndrome HELLP considerando esses achados?")*
>
> "As perguntas de follow-up têm o contexto completo da sessão — o agente sabe os dados da paciente e o diagnóstico anterior."

---

## BLOCO 8 — Observabilidade e Testes (1 min)

**O que mostrar na tela:** terminal com `tail -f logs/app.log` e depois `pytest tests/ -v`

**Fala:**

> "Cada sessão gera logs estruturados em JSON — um evento por linha. Aqui dá pra ver o `session_id` correlacionando todos os eventos de uma mesma sessão: o início, a coleta de cada campo, a predição do modelo com o nível de risco e as probabilidades, e cada chamada ao Claude com o número de tokens consumidos e a latência em milissegundos."
>
> "O design foi feito para ser plugável: o logger local escreve em arquivo e em stdout. Para adicionar observabilidade em nuvem — Datadog, CloudWatch, GCP Logging — basta adicionar um handler aqui em `setup_logging()`. Nenhuma linha de código de negócio precisa mudar."
>
> "Importante: os dados clínicos da paciente não são logados — apenas metadados da sessão e métricas de performance."
>
> *(mudar para pytest)*
>
> "21 testes automatizados em 4 arquivos. `test_encoding.py` valida a codificação dos genes e os bounds. `test_operators.py` valida seleção, cruzamento e mutação — inclusive que o cruzamento uniforme sempre produz genes dentro dos bounds dos pais. `test_ga.py` valida o loop principal e o elitismo. `test_prompts.py` valida a construção dos prompts dos agentes. Todos passando."

---

## BLOCO 9 — Encerramento (30 seg)

**O que mostrar na tela:** README no GitHub

**Fala:**

> "O repositório está no GitHub com todo o código, os notebooks executados, os resultados dos experimentos e o relatório técnico. Para rodar: clonar, criar o ambiente virtual, instalar as dependências, configurar a chave da API Anthropic no .env e rodar `streamlit run app.py`.
>
> Obrigado."

---

## Dicas de Gravação

- **Antes de gravar:** deixar o app já rodando (`streamlit run app.py`) e o terminal com `tail -f logs/app.log` aberto numa segunda janela
- **Resolução:** 1920×1080
- **Fonte do VS Code:** aumentar para pelo menos 16pt para legibilidade no vídeo
- **Paciente de demo sugerida para alto risco:** Idade=48, Sistólica=160, Diastólica=100, Glicemia=15.0, Temperatura=38.5°C, FC=105 — garante classificação de alto risco com confiança alta
- **Paciente de demo sugerida para baixo risco:** Idade=25, Sistólica=90, Diastólica=60, Glicemia=6.0, Temperatura=36.8°C, FC=76
- **Não mostrar o arquivo `.env` nem a chave de API em nenhum momento**
