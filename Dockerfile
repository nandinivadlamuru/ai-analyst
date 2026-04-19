# Optional: run the Streamlit UI in a container. Ollama stays on the host —
# set OLLAMA_URL to host.docker.internal:11434 on Mac/Windows (see README).
FROM python:3.11-slim

WORKDIR /app

RUN apt-get update && apt-get install -y --no-install-recommends \
    build-essential \
    && rm -rf /var/lib/apt/lists/*

COPY pyproject.toml README.md ./
COPY src ./src

RUN pip install --no-cache-dir pip setuptools wheel \
    && pip install --no-cache-dir -e .

COPY data/sample ./data/sample
COPY eval ./eval

ENV PYTHONUNBUFFERED=1
EXPOSE 8501

CMD ["streamlit", "run", "src/grounded_analyst/ui.py", "--server.address", "0.0.0.0", "--server.port", "8501"]
