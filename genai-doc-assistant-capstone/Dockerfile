FROM python:3.11-slim

WORKDIR /app

# Install system dependencies required for pycorekit build
RUN apt-get update && apt-get install -y --no-install-recommends \
    build-essential \
    python3-dev \
    libssl-dev \
    rustc \
    cargo \
    git \
    curl \
    && rm -rf /var/lib/apt/lists/*

# Copy backend project
COPY . /app

# Install Python dependencies
RUN pip install --upgrade pip
RUN pip install --no-cache-dir -r requirements.txt

# Download NLTK data
RUN python -m nltk.downloader stopwords

EXPOSE 8000

# Install pycorekit at container startup (mounted at runtime)
CMD pip install /app/pycorekit && \
    uvicorn main:app --host 0.0.0.0 --port 8000
