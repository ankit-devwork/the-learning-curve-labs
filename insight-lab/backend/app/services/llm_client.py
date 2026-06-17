import hashlib
import os
import re

import litellm

from app.core.config import settings
from app.core.exceptions import ServiceUnavailableException
from app.core.yaml_config import get_yaml_config


def _llm_model() -> str:
    return os.getenv("LITELLM_MODEL") or get_yaml_config().llm.model


def _ensure_llm_configured() -> None:
    if not settings.groq_api_key.strip():
        raise ServiceUnavailableException("GROQ_API_KEY is not configured on the backend")


async def generate_summary(document_text: str, *, filename: str) -> str:
    _ensure_llm_configured()
    llm = get_yaml_config().llm
    prompt = (
        "Summarize the following document clearly and concisely for a learner. "
        "Use short sections and bullet points where helpful.\n\n"
        f"Filename: {filename}\n\n"
        f"Document:\n{document_text[:12000]}"
    )
    response = await litellm.acompletion(
        model=_llm_model(),
        messages=[
            {"role": "system", "content": "You are a helpful document analyst."},
            {"role": "user", "content": prompt},
        ],
        max_tokens=llm.summary_max_tokens,
        api_key=settings.groq_api_key,
    )
    return response.choices[0].message.content.strip()


async def answer_question(
    *,
    question: str,
    context_chunks: list[str],
    filename: str,
) -> dict:
    _ensure_llm_configured()
    llm = get_yaml_config().llm
    context = "\n\n---\n\n".join(
        f"[Chunk {index + 1}]\n{chunk}" for index, chunk in enumerate(context_chunks)
    )
    prompt = (
        "Answer the user's question using ONLY the provided document excerpts. "
        "If the answer is not in the excerpts, say you don't know based on this document. "
        "Keep the answer concise and cite chunk numbers like [Chunk 2] when relevant.\n\n"
        f"Document: {filename}\n\n"
        f"Excerpts:\n{context}\n\n"
        f"Question: {question}"
    )
    response = await litellm.acompletion(
        model=_llm_model(),
        messages=[
            {"role": "system", "content": "You are a grounded document Q&A assistant."},
            {"role": "user", "content": prompt},
        ],
        max_tokens=llm.chat_max_tokens,
        api_key=settings.groq_api_key,
    )
    answer = response.choices[0].message.content.strip()
    cited = sorted({int(n) for n in re.findall(r"\[Chunk (\d+)\]", answer)})
    return {"answer": answer, "cited_chunks": cited}


def question_cache_key(document_id: str, question: str) -> str:
    digest = hashlib.sha256(question.strip().lower().encode("utf-8")).hexdigest()[:16]
    return f"chat:document:{document_id}:{digest}"
