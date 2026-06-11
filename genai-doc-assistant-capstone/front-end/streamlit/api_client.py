import os
import requests

BASE_URL = os.getenv("BACKEND_URL", "http://backend:8000").rstrip("/")


class BackendAPIError(Exception):
    def __init__(self, message: str, status_code: int | None = None, correlation_id: str | None = None):
        super().__init__(message)
        self.status_code = status_code
        self.correlation_id = correlation_id


class BackendClient:
    """HTTP client for the FastAPI backend."""

    def __init__(self, base_url: str | None = None):
        self.base_url = (base_url or BASE_URL).rstrip("/")

    def _handle_response(self, resp: requests.Response) -> dict:
        correlation_id = resp.headers.get("x-correlation-id")
        try:
            data = resp.json()
        except ValueError:
            data = {"error": resp.text or "Invalid JSON response"}

        if resp.status_code >= 400:
            message = data.get("error") or data.get("message") or resp.text
            raise BackendAPIError(
                message=message,
                status_code=resp.status_code,
                correlation_id=data.get("correlation_id") or correlation_id,
            )

        if isinstance(data, dict) and "correlation_id" not in data and correlation_id:
            data["correlation_id"] = correlation_id
        return data

    def upload_document(self, file) -> dict:
        files = {"file": (file.name, file.getvalue())}
        resp = requests.post(f"{self.base_url}/upload-and-ingest", files=files, timeout=300)
        return self._handle_response(resp)

    def ask_question(self, question: str, thread_id: str) -> dict:
        payload = {"question": question, "thread_id": thread_id}
        resp = requests.post(f"{self.base_url}/ask-question", json=payload, timeout=300)
        return self._handle_response(resp)

    def choose_document(self, thread_id: str, question: str, selected_doc_id: str) -> dict:
        payload = {
            "thread_id": thread_id,
            "question": question,
            "selected_doc_id": selected_doc_id,
        }
        resp = requests.post(f"{self.base_url}/choose-document", json=payload, timeout=300)
        return self._handle_response(resp)

    def list_documents(self) -> dict:
        resp = requests.get(f"{self.base_url}/documents", timeout=60)
        return self._handle_response(resp)

    def readiness(self) -> dict:
        resp = requests.get(f"{self.base_url}/ready", timeout=30)
        return self._handle_response(resp)
