FROM python:3.11-slim

WORKDIR /app

# System deps needed by pymupdf / sentence-transformers wheels on slim images.
RUN apt-get update && apt-get install -y --no-install-recommends \
    build-essential \
    && rm -rf /var/lib/apt/lists/*

COPY requirements.txt .
# --retries/--timeout raised above pip's defaults to ride out slow/unreliable
# networks during image builds; harmless on a fast connection.
RUN pip install --no-cache-dir --retries 10 --timeout 100 -r requirements.txt

COPY . .

# Secrets (OPENAI_API_KEY, OPENROUTER_API_KEY) are provided at runtime via
# env_file in docker-compose.yml — never baked into the image.

EXPOSE 8000

CMD ["uvicorn", "app:app", "--host", "0.0.0.0", "--port", "8000"]
