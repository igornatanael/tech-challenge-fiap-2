FROM python:3.11-slim

WORKDIR /app

# Dependências do sistema necessárias para compilar pacotes científicos
RUN apt-get update && apt-get install -y --no-install-recommends \
    build-essential \
    && rm -rf /var/lib/apt/lists/*

COPY requirements.txt .

# Instala dependências exceto streamlit (não necessário no container de treino/teste)
RUN grep -v "^streamlit" requirements.txt > requirements_server.txt \
    && pip install --no-cache-dir -r requirements_server.txt \
    && rm requirements_server.txt

# Copia código e dados
COPY src/ ./src/
COPY data/ ./data/
COPY tests/ ./tests/
COPY experiments/ ./experiments/

# Usuário não-root para segurança
RUN useradd -m appuser && chown -R appuser /app
USER appuser

# A chave de API deve ser injetada em tempo de execução via variável de ambiente:
#   docker run -e ANTHROPIC_API_KEY=sk-ant-... <image>
# Nunca incluir a chave no build ou no .env copiado para a imagem.

CMD ["pytest", "tests/", "-v", "--tb=short"]
