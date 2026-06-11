# Build from monorepo root:
#   docker build -f genai-doc-assistant-capstone/Dockerfile -t genai-backend .

FROM python:3.11-slim AS builder

WORKDIR /build

ENV PATH=/root/.local/bin:$PATH

RUN apt-get update && apt-get install -y --no-install-recommends \
    build-essential \
    python3-dev \
    libssl-dev \
    && rm -rf /var/lib/apt/lists/*

COPY genai-doc-assistant-capstone/requirements.txt ./requirements.txt
# Install CPU torch first so sentence-transformers does not pull CUDA wheels.
RUN pip install --upgrade pip && \
    pip install --user --no-cache-dir torch --index-url https://download.pytorch.org/whl/cpu && \
    pip install --user --no-cache-dir -r requirements.txt

COPY pycorekit /build/pycorekit
RUN pip install --user --no-cache-dir /build/pycorekit

FROM python:3.11-slim AS runtime

WORKDIR /app

RUN apt-get update && apt-get install -y --no-install-recommends \
    curl \
    && rm -rf /var/lib/apt/lists/*

COPY --from=builder /root/.local /root/.local

ENV PATH=/root/.local/bin:$PATH
ENV PYTHONUNBUFFERED=1
ENV PYTHONDONTWRITEBYTECODE=1

COPY genai-doc-assistant-capstone /app

RUN mkdir -p /app/data/uploads /app/data/vector_store /app/logs && \
    chmod +x /app/scripts/start-backend.sh

EXPOSE 8000

HEALTHCHECK --interval=15s --timeout=5s --start-period=60s --retries=5 \
    CMD sh -c 'curl -f "http://localhost:${PORT:-8000}/health" || exit 1'

CMD ["/app/scripts/start-backend.sh"]
