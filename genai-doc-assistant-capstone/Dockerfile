FROM python:3.11-slim

WORKDIR /app

RUN apt-get update && apt-get install -y --no-install-recommends \
    build-essential \
    python3-dev \
    libssl-dev \
    curl \
    && rm -rf /var/lib/apt/lists/*

COPY requirements.txt /app/requirements.txt
RUN pip install --upgrade pip && \
    pip install --no-cache-dir -r requirements.txt && \
    pip install torch --index-url https://download.pytorch.org/whl/cpu

COPY . /app

EXPOSE 8000

CMD pip install /app/pycorekit && \
    uvicorn main:app --host 0.0.0.0 --port 8000
