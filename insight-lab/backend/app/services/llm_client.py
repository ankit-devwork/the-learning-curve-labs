import hashlib
import json
import os
import re

import litellm

from app.core.config import settings
from app.core.exceptions import ServiceUnavailableException
from app.core.resilience import llm_circuit, with_retry
from app.core.yaml_config import get_yaml_config


def _llm_model() -> str:
    return os.getenv("LITELLM_MODEL") or get_yaml_config().llm.model


def _ensure_llm_configured() -> None:
    if not settings.groq_api_key.strip():
        raise ServiceUnavailableException("GROQ_API_KEY is not configured on the backend")


async def _acompletion_with_resilience(*, messages: list[dict], max_tokens: int) -> str:
    _ensure_llm_configured()

    async def _call() -> str:
        async def _invoke() -> str:
            response = await litellm.acompletion(
                model=_llm_model(),
                messages=messages,
                max_tokens=max_tokens,
                api_key=settings.groq_api_key,
            )
            return response.choices[0].message.content.strip()

        return await with_retry(_invoke, operation="llm.acompletion")

    return await llm_circuit.call(_call)


async def generate_summary(document_text: str, *, filename: str) -> str:
    llm = get_yaml_config().llm
    prompt = (
        "Summarize the following document clearly and concisely for a learner. "
        "Use short sections and bullet points where helpful.\n\n"
        f"Filename: {filename}\n\n"
        f"Document:\n{document_text[:12000]}"
    )
    return await _acompletion_with_resilience(
        messages=[
            {"role": "system", "content": "You are a helpful document analyst."},
            {"role": "user", "content": prompt},
        ],
        max_tokens=llm.summary_max_tokens,
    )


async def answer_question(
    *,
    question: str,
    context_chunks: list[str],
    filename: str,
) -> dict:
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
    answer = await _acompletion_with_resilience(
        messages=[
            {"role": "system", "content": "You are a grounded document Q&A assistant."},
            {"role": "user", "content": prompt},
        ],
        max_tokens=llm.chat_max_tokens,
    )
    cited = sorted({int(n) for n in re.findall(r"\[Chunk (\d+)\]", answer)})
    return {"answer": answer, "cited_chunks": cited}


async def generate_excel_chart_plan(*, profile: dict, filename: str) -> str:
    cfg = get_yaml_config().excel
    prompt = (
        "You are a data analyst. Given spreadsheet metadata, return ONLY valid JSON "
        "with a 'charts' array (max "
        f"{cfg.max_charts} items). Each chart object must include: "
        "id, title, chart_type (bar|line|pie|scatter), x_column, y_column (optional), "
        "aggregation (sum|mean|count|none). Pick meaningful charts for the data.\n\n"
        f"Filename: {filename}\n\n"
        f"Profile:\n{json.dumps(profile, indent=2)[:12000]}"
    )
    return await _acompletion_with_resilience(
        messages=[
            {"role": "system", "content": "Return JSON only. No markdown."},
            {"role": "user", "content": prompt},
        ],
        max_tokens=cfg.chart_plan_max_tokens,
    )


async def generate_excel_summary(*, profile: dict, charts: list[dict], filename: str) -> str:
    cfg = get_yaml_config().excel
    prompt = (
        "Write a concise narrative summary of insights from this spreadsheet analysis. "
        "Reference the charts by title. Use bullet points for key findings.\n\n"
        f"Filename: {filename}\n\n"
        f"Profile:\n{json.dumps(profile, indent=2)[:6000]}\n\n"
        f"Charts:\n{json.dumps(charts, indent=2)[:6000]}"
    )
    return await _acompletion_with_resilience(
        messages=[
            {"role": "system", "content": "You are a helpful data analyst."},
            {"role": "user", "content": prompt},
        ],
        max_tokens=cfg.summary_max_tokens,
    )


async def generate_quiz_draft(
    *,
    context_chunks: list[str],
    filename: str,
    question_type: str,
    difficulty: str,
    num_questions: int,
) -> str:
    cfg = get_yaml_config().quizzes
    context = "\n\n---\n\n".join(
        f"[Chunk {index + 1}]\n{chunk}" for index, chunk in enumerate(context_chunks)
    )
    prompt = (
        "Create a quiz from the document excerpts below. Return ONLY valid JSON with:\n"
        "- title (string)\n"
        "- questions (array)\n\n"
        "Each question object must include:\n"
        "- question_text (string)\n"
        "- options (array of 2-6 strings)\n"
        "- correct_option_index (0-based integer)\n"
        "- explanation (string)\n"
        "- source_chunk_index (0-based index into the provided chunks)\n\n"
        f"Quiz settings: question_type={question_type}, difficulty={difficulty}, "
        f"num_questions={num_questions}\n"
        "Use only facts supported by the excerpts. For true_false use exactly two options: True and False.\n\n"
        f"Document: {filename}\n\n"
        f"Excerpts:\n{context[:12000]}"
    )
    return await _acompletion_with_resilience(
        messages=[
            {"role": "system", "content": "Return JSON only. No markdown."},
            {"role": "user", "content": prompt},
        ],
        max_tokens=cfg.quiz_max_tokens,
    )


def question_cache_key(document_id: str, question: str) -> str:
    digest = hashlib.sha256(question.strip().lower().encode("utf-8")).hexdigest()[:16]
    return f"chat:document:{document_id}:{digest}"


def excel_cache_key(document_id: str, file_hash: str | None) -> str:
    digest = file_hash or "unknown"
    return f"excel:charts:{document_id}:{digest}"


def quiz_cache_key(document_id: str, settings_hash: str) -> str:
    return f"quiz:{document_id}:{settings_hash}"
