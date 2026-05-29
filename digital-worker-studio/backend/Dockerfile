# STAGE 1: Build dependencies
FROM python:3.11-slim AS builder

WORKDIR /build

# Install system dependencies for database drivers
RUN apt-get update && apt-get install -y --no-install-recommends \
    build-essential \
    libpq-dev \
    && rm -rf /var/lib/apt/lists/*

# Copy requirements.txt from the ROOT context
COPY requirements.txt .
RUN pip install --no-cache-dir --user -r requirements.txt

# STAGE 2: Lightweight runtime
FROM python:3.11-slim AS runtime

WORKDIR /app

# Install runtime postgres libraries
RUN apt-get update && apt-get install -y --no-install-recommends \
    libpq5 \
    && rm -rf /var/lib/apt/lists/*

# Pull packages from builder
COPY --from=builder /root/.local /root/.local

# Copy application source from backend/ directory
COPY backend/ .

# 🚀 FIX: Explicitly pull config.properties from the root build context into the active working directory (/app)
COPY config.properties ./config.properties

# Set environment paths
ENV PATH=/root/.local/bin:$PATH
ENV PYTHONPATH=/app
ENV PYTHONUNBUFFERED=1

EXPOSE 8000

CMD ["uvicorn", "app.main:app", "--host", "0.0.0.0", "--port", "8000"]