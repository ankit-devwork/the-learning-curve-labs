# Build from monorepo root:
#   docker build -f genai-doc-assistant-capstone/front-end/streamlit/Dockerfile -t genai-streamlit .

FROM python:3.11-slim

WORKDIR /app

RUN apt-get update && apt-get install -y --no-install-recommends \
    curl \
    && rm -rf /var/lib/apt/lists/*

COPY genai-doc-assistant-capstone/front-end/streamlit/requirements.txt ./requirements.txt
RUN pip install --upgrade pip && \
    pip install --no-cache-dir -r requirements.txt

COPY genai-doc-assistant-capstone/front-end/streamlit /app

ENV PYTHONUNBUFFERED=1

RUN chmod +x /app/scripts/start-streamlit.sh

EXPOSE 8501

HEALTHCHECK --interval=20s --timeout=5s --start-period=15s --retries=5 \
    CMD sh -c 'curl -f "http://localhost:${PORT:-8501}/_stcore/health" || exit 1'

CMD ["/app/scripts/start-streamlit.sh"]
