FROM python:3.12-slim

# Be patient with slow/flaky package mirrors when building on the GPU box.
ENV PIP_DEFAULT_TIMEOUT=180 \
    PIP_RETRIES=10

WORKDIR /app

# OCR engine for scanned textbooks (opt-in via OCR_ENABLED; ingest-time only).
# The tesseract binary + Russian/Kazakh language data; PDF rendering itself is a
# pure pypdfium2 wheel, so no poppler is needed. Slim layer: drop apt lists.
RUN apt-get update \
    && apt-get install -y --no-install-recommends \
        tesseract-ocr tesseract-ocr-rus tesseract-ocr-kaz \
    && rm -rf /var/lib/apt/lists/*

COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

COPY app ./app
COPY scripts ./scripts
COPY scenarios ./scenarios

EXPOSE 8000

# Stateless proxy — a single worker is plenty; scale horizontally if needed.
CMD ["uvicorn", "app.main:app", "--host", "0.0.0.0", "--port", "8000", "--workers", "1"]
