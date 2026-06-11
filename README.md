# the-learning-curve-labs

Code, experiments, and hands‑on demos from The Learning Curve newsletter — covering AI/ML concepts, PoC implementations, semantic search, embeddings, and practical machine learning workflows.

## Projects

| Folder | Description |
|--------|-------------|
| [genai-doc-assistant-capstone/](genai-doc-assistant-capstone/) | Multi-agent RAG document assistant (FastAPI + Streamlit) |
| [pycorekit/](pycorekit/) | Shared logging, tracing, and config utilities |
| [digital-worker-studio/](digital-worker-studio/) | AI workflow automation platform |

### Run the capstone app (Docker)

```bash
cp genai-doc-assistant-capstone/.env.example genai-doc-assistant-capstone/.env
# Set GROQ_API_KEY in .env

docker compose -f genai-doc-assistant-capstone/docker-compose.yml up --build
```

See [genai-doc-assistant-capstone/docs/DOCKER.md](genai-doc-assistant-capstone/docs/DOCKER.md) for details.
