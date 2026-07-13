FROM python:3.11-slim

WORKDIR /app

# Install dependencies
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Copy source
COPY src/ ./src/
COPY data/ ./data/
COPY notebooks/ ./notebooks/
COPY experiments/ ./experiments/

# Copy environment example (user must provide .env at runtime)
COPY .env.example .env.example

# Default command: run tests
CMD ["pytest", "tests/", "-v"]
