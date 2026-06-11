"""
db_connection.py
----------------
Centralized database layer for vector storage and chat history.

Responsibilities:
- Initialize ChromaDB persistent client
- Provide shared collection instances
- Provide embedding model (cached)
- Expose helper functions for:
    • Adding/querying vector embeddings
    • Storing and retrieving chat history (no Redis/Postgres required)
"""

from pathlib import Path
from functools import lru_cache
from typing import List, Dict, Any
from datetime import datetime, timezone
import uuid
import asyncio

from chromadb import Client
from chromadb.config import Settings
from sentence_transformers import SentenceTransformer

from pycorekit.logging.logger import get_logger

# Unified tracing engine
from pycorekit.tracing.tracing import start_trace

# Structured DB observability
from pycorekit.observability.observability import observe_db

from pycorekit.correlation.context import get_current_correlation_id
from app.core.settings import settings

log = get_logger("db")


# ---------------------------------------------------------------------------
# Embedding Model (cached)
# ---------------------------------------------------------------------------

@lru_cache(maxsize=1)
def get_embedding_model():
    """
    Loads the embedding model once and caches it.
    """
    with start_trace("load_embedding_model", inputs={"model": settings.models.embedding_model}):
        log.info(f"Loading embedding model: {settings.models.embedding_model}")
        return SentenceTransformer(settings.models.embedding_model)


def embed_texts(texts: List[str], batch_size: int = 32) -> List[List[float]]:
    """
    Embed a list of text chunks into vector embeddings (sync).
    """
    with start_trace("embed_texts_sync", inputs={"num_texts": len(texts)}):
        model = get_embedding_model()
        embeddings = []

        for i in range(0, len(texts), batch_size):
            batch = texts[i:i + batch_size]
            emb = model.encode(batch, convert_to_numpy=True)
            embeddings.extend(emb)

        return [e.tolist() for e in embeddings]


async def async_embed_texts(texts: List[str], batch_size: int = 32) -> List[List[float]]:
    """
    Async wrapper around embed_texts using asyncio.to_thread.
    """
    with start_trace("embed_texts_async", inputs={"num_texts": len(texts)}):
        return await asyncio.to_thread(embed_texts, texts, batch_size)


# ---------------------------------------------------------------------------
# Chroma Client (persistent)
# ---------------------------------------------------------------------------

@lru_cache(maxsize=1)
def get_chroma_client():
    """
    Returns a persistent ChromaDB client using the NEW API.
    """
    persist_dir = Path(settings.paths.vector_db_path)

    with start_trace("get_chroma_client", inputs={"path": str(persist_dir)}):
        log.info(f"Initializing ChromaDB (new API) at: {persist_dir}")

        return Client(
            Settings(
                chroma_api_impl="chromadb.api.rust.RustBindingsAPI",
                persist_directory=str(persist_dir),
                anonymized_telemetry=False
            )
        )


def get_collection(name: str = "documents"):
    """
    Returns (or creates) a Chroma collection by name.
    MUST remain synchronous.
    MUST NOT use observe_db.
    """
    with start_trace("get_collection", inputs={"name": name}):
        client = get_chroma_client()
        return client.get_or_create_collection(name=name)


async def async_get_collection(name: str = "documents"):
    """
    Async wrapper around get_collection.
    """
    with start_trace("get_collection_async", inputs={"name": name}):
        return await asyncio.to_thread(get_collection, name)


# ---------------------------------------------------------------------------
# Vector Storage (Document Chunks)
# ---------------------------------------------------------------------------

def add_to_chroma(
    doc_id: str,
    chunks: List[str],
    embeddings: List[List[float]],
    metadata: List[Dict[str, Any]],
    collection_name: str = "documents",
):
    """
    Adds chunks + embeddings + metadata to ChromaDB (sync).
    """
    with start_trace("add_to_chroma_sync", inputs={"doc_id": doc_id, "num_chunks": len(chunks)}):
        collection = get_collection(collection_name)
        ids = [f"{doc_id}_{i}" for i in range(len(chunks))]

        log.info(
            f"Adding {len(chunks)} chunks to Chroma",
            extra={"doc_id": doc_id, "collection": collection_name},
        )

        return collection.add(
            ids=ids,
            documents=chunks,
            embeddings=embeddings,
            metadatas=metadata,
        )


async def async_add_to_chroma(
    doc_id: str,
    chunks: List[str],
    embeddings: List[List[float]],
    metadata: List[Dict[str, Any]],
    collection_name: str = "documents",
):
    """
    Async wrapper around add_to_chroma using asyncio.to_thread.
    """
    with start_trace("add_to_chroma_async", inputs={"doc_id": doc_id, "num_chunks": len(chunks)}):
        return await observe_db(
            "chroma_add_async",
            func=lambda: asyncio.to_thread(
                add_to_chroma,
                doc_id, chunks, embeddings, metadata, collection_name
            ),
            query=f"add_async {len(chunks)} chunks to {collection_name}"
        )


# ---------------------------------------------------------------------------
# Chat History Storage (Chroma-based)
# ---------------------------------------------------------------------------

def get_chat_collection():
    """
    Returns the chat_history collection.
    MUST NOT wrap get_collection() in observe_db.
    """
    with start_trace("get_chat_collection"):
        return get_collection("chat_history")


def save_chat_message(thread_id: str, role: str, content: str):
    """
    Save a single chat message to Chroma.
    Uses dummy embeddings (no semantic search needed).
    """
    with start_trace("save_chat_message", inputs={"thread_id": thread_id, "role": role}):
        collection = get_chat_collection()
        msg_id = f"{thread_id}_{uuid.uuid4()}"

        embedding_dim = getattr(settings.models, "embedding_dim", 768)
        dummy_embedding = [0.0] * embedding_dim

        return collection.add(
            ids=[msg_id],
            documents=[content],
            embeddings=[dummy_embedding],
            metadatas=[{
                "thread_id": thread_id,
                "role": role,
                "content": content,
                "timestamp": datetime.now(timezone.utc).isoformat()
            }]
        )


async def async_save_chat_message(thread_id: str, role: str, content: str):
    """
    Async wrapper around save_chat_message.
    """
    with start_trace("save_chat_message_async", inputs={"thread_id": thread_id, "role": role}):
        return await observe_db(
            "chroma_add_chat_message_async",
            func=lambda: asyncio.to_thread(save_chat_message, thread_id, role, content),
            query=f"add chat message async thread={thread_id}"
        )


def load_chat_history(thread_id: str) -> List[Dict[str, str]]:
    """
    Load chat history for a given thread_id (ordered by timestamp).
    """
    with start_trace("load_chat_history", inputs={"thread_id": thread_id}):
        collection = get_chat_collection()

        results = collection.get(
            where={"thread_id": thread_id},
            include=["metadatas"]
        )

        if not results or not results.get("metadatas"):
            return []

        history = sorted(results["metadatas"], key=lambda x: x.get("timestamp", ""))

        return [{"role": h["role"], "content": h["content"]} for h in history]


async def async_load_chat_history(thread_id: str) -> List[Dict[str, str]]:
    """
    Async wrapper around load_chat_history.
    """
    with start_trace("load_chat_history_async", inputs={"thread_id": thread_id}):
        return await observe_db(
            "chroma_get_chat_history_async",
            func=lambda: asyncio.to_thread(load_chat_history, thread_id),
            query=f"get chat history async thread={thread_id}"
        )
