FROM python:3.11-slim

WORKDIR /app

# Install system dependencies (optional but recommended)
RUN apt-get update && apt-get install -y curl && apt-get clean

# Install Python dependencies
RUN pip install --no-cache-dir streamlit requests plotly pandas

# Copy app code
COPY . /app

# Expose Streamlit port
EXPOSE 8501

# Streamlit entrypoint
CMD ["streamlit", "run", "chat.py", "--server.port=8501", "--server.address=0.0.0.0"]