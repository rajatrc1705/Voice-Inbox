FROM python:3.12-slim

ENV PYTHONUNBUFFERED=1 \
    PYTHONDONTWRITEBYTECODE=1

WORKDIR /app

COPY pyproject.toml /app/pyproject.toml
COPY agent.py /app/agent.py
COPY api /app/api
RUN python -m venv /app/.venv && \
    /app/.venv/bin/pip install .

CMD ["/bin/sh", "-c", "exec /app/.venv/bin/uvicorn api.main:app --host 0.0.0.0 --port ${PORT:-8080}"]
