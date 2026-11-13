FROM python:3.12-slim

WORKDIR /app

COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

COPY src/ src/
COPY data/v1/chunks.jsonl data/v1/filings.json data/v1/

ENV BASE_PATH=/app

CMD uvicorn app:app --app-dir src --host 0.0.0.0 --port ${PORT:-8080}