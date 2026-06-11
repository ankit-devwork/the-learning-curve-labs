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

EXPOSE 8501

HEALTHCHECK --interval=20s --timeout=5s --start-period=15s --retries=5 \
    CMD curl -f http://localhost:8501/_stcore/health || exit 1

CMD ["streamlit", "run", "chat.py", "--server.port=8501", "--server.address=0.0.0.0", "--server.headless=true", "--browser.gatherUsageStats=false"]
