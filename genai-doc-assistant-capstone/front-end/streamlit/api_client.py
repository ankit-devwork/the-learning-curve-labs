import os
import requests

# ---------------------------------------------------------
# Backend URL (Docker + Local compatible)
# ---------------------------------------------------------
BASE_URL = os.getenv("BACKEND_URL", "http://backend:8000").rstrip("/")


class BackendClient:
    """
    Simple HTTP client for interacting with the FastAPI backend.
    Wraps all endpoints:
        - upload-and-ingest
        - ask-question
        - choose-document
        - documents
    """

    def __init__(self, base_url: str = None):
        self.base_url = (base_url or BASE_URL).rstrip("/")

    # ---------------------------------------------------------
    # 1. Upload + Ingest Document
    # ---------------------------------------------------------
    def upload_document(self, file):
        """
        Upload a document and ingest it into ChromaDB.
        Streamlit file uploader gives a file-like object.
        """
        files = {"file": (file.name, file.getvalue())}

        try:
            resp = requests.post(f"{self.base_url}/upload-and-ingest", files=files)
            resp.raise_for_status()
            return resp.json()
        except Exception as e:
            return {"error": str(e)}

    # ---------------------------------------------------------
    # 2. Ask Question
    # ---------------------------------------------------------
    def ask_question(self, question: str, thread_id: str):
        payload = {"question": question, "thread_id": thread_id}

        try:
            resp = requests.post(f"{self.base_url}/ask-question", json=payload)
            resp.raise_for_status()
            return resp.json()
        except Exception as e:
            return {"error": str(e)}

    # ---------------------------------------------------------
    # 3. Choose Document (HITL)
    # ---------------------------------------------------------
    def choose_document(self, thread_id: str, question: str, selected_doc_id: str):
        payload = {
            "thread_id": thread_id,
            "question": question,
            "selected_doc_id": selected_doc_id,
        }

        try:
            resp = requests.post(f"{self.base_url}/choose-document", json=payload)
            resp.raise_for_status()
            return resp.json()
        except Exception as e:
            return {"error": str(e)}

    # ---------------------------------------------------------
    # 4. List Documents
    # ---------------------------------------------------------
    def list_documents(self):
        try:
            resp = requests.get(f"{self.base_url}/documents")
            resp.raise_for_status()
            return resp.json()
        except Exception as e:
            return {"error": str(e)}
