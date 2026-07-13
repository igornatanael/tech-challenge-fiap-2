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

> "A estrutura está organizada assim: dentro de `src` temos os módulos separados por responsabilidade — pipelines de dados, modelos baseline, algoritmo genético e a camada de LLM com os agentes. Os notebooks documentam cada etapa em ordem. Os resultados dos experimentos ficam em `experiments` como JSON. E o `app.py` é a interface web."

---

## BLOCO 3 — Algoritmo Genético: Implementação (2 min)

**O que mostrar na tela:** `src/genetic_algorithm/encoding.py` → `operators.py` → `ga.py`

**Fala:**

> "O Algoritmo Genético otimiza os hiperparâmetros do Random Forest. Cada indivíduo na população é um dicionário com 5 genes — n_estimators, max_depth, min_samples_split, min_samples_leaf e max_features — com bounds e tipos definidos aqui no SEARCH_SPACES."
>
> *(abrir operators.py)*
>
> "Os operadores são: seleção por torneio — pegamos 3 candidatos aleatórios e o melhor vira pai. Cruzamento uniforme — cada gene do filho é herdado de um dos pais com probabilidade 0.5. Mutação por gene — com probabilidade configurável, o gene recebe um novo valor dentro dos bounds."
>
> *(abrir ga.py, mostrar o loop)*
>
> "O loop principal usa elitismo — o melhor indivíduo global nunca é perdido. A função fitness avalia cada indivíduo com StratifiedKFold de 5 folds, usando F1-macro como critério — importante porque o problema é multiclasse e a classe de alto risco é crítica."

---

## BLOCO 4 — Algoritmo Genético: Resultados (2 min)

**O que mostrar na tela:** Notebook 02 executado — curvas de convergência e tabela comparativa

**Fala:**

> "Rodamos 3 experimentos com configurações diferentes. Aqui estão as curvas de convergência — cada linha é um experimento, a linha vermelha pontilhada é o baseline do GridSearch."
>
> *(rolar até a tabela comparativa)*
>
> "O Experimento 1 — configuração padrão, população 30, 20 gerações — foi o melhor. F1-macro subiu de 0.9017 para 0.9070, e o recall de alto risco, que é a métrica mais importante clinicamente, subiu de 0.9487 para 0.9744."
>
> "O que o AG encontrou de diferente do GridSearch? Árvores menores: max_depth 15 em vez de irrestrito, n_estimators 73 em vez de 100. A validação cruzada penalizou overfitting de forma mais eficaz do que a grade estática. Os resultados ficaram em JSON aqui em `experiments/`."

---

## BLOCO 5 — Arquitetura LLM (1,5 min)

**O que mostrar na tela:** `src/llm/agents/` — mostrar os arquivos, abrir `base_agent.py` e depois `patient_agent.py` e `doctor_agent.py` lado a lado

**Fala:**

> "Para a integração com LLM, a arquitetura usa dois agentes especializados que herdam de uma classe base. O BaseAgent mantém o histórico de conversa multi-turn — cada chamada a `chat()` inclui todo o histórico, permitindo que o agente mantenha contexto ao longo de uma sessão."
>
> "O PatientAgent usa linguagem acessível, sem jargão, e tem guardrails explícitos — não prescreve medicamentos, não extrapola o escopo. O DoctorAgent usa terminologia clínica, inclui valores de referência no system prompt e estrutura a resposta em 4 seções: análise do modelo, avaliação clínica, investigação sugerida e conduta."
>
> "Além disso, temos um avaliador — LLM-as-judge — que usa o próprio Claude para avaliar a qualidade das respostas com rubrics distintos por perfil."

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

> "Cada sessão gera logs estruturados em JSON. Aqui dá pra ver o session_id correlacionando todos os eventos — início da sessão, coleta dos dados, predição do modelo e cada chamada ao Claude com tokens consumidos e latência. O design permite plugar qualquer ferramenta de observabilidade cloud sem mudar o código."
>
> *(mudar para pytest)*
>
> "21 testes automatizados cobrindo a codificação genética, os operadores do AG, o loop principal e a construção dos prompts. Todos passando."

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
