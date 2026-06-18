import json
from typing import Any

from pydantic import BaseModel, Field, field_validator, model_validator


class QuizQuestionDraft(BaseModel):
    question_text: str
    options: list[str] = Field(min_length=2, max_length=6)
    correct_option_index: int = Field(ge=0)
    explanation: str | None = None
    source_chunk_index: int | None = Field(default=None, ge=0)

    @field_validator("options")
    @classmethod
    def strip_options(cls, value: list[str]) -> list[str]:
        cleaned = [option.strip() for option in value if option.strip()]
        if len(cleaned) < 2:
            raise ValueError("Each question needs at least two options")
        return cleaned

    @model_validator(mode="after")
    def validate_correct_index(self) -> "QuizQuestionDraft":
        if self.correct_option_index >= len(self.options):
            raise ValueError("correct_option_index is out of range")
        return self


class QuizDraft(BaseModel):
    title: str
    questions: list[QuizQuestionDraft] = Field(min_length=1)


def parse_quiz_draft(raw: str, *, max_questions: int) -> QuizDraft:
    text = raw.strip()
    if text.startswith("```"):
        text = text.split("\n", 1)[-1]
        if text.endswith("```"):
            text = text.rsplit("```", 1)[0]
    payload = json.loads(text)
    if isinstance(payload, dict) and "questions" in payload:
        draft = QuizDraft.model_validate(payload)
    elif isinstance(payload, list):
        draft = QuizDraft.model_validate({"title": "Document quiz", "questions": payload})
    else:
        raise ValueError("Quiz payload must include a 'questions' array")
    draft.questions = draft.questions[:max_questions]
    if not draft.questions:
        raise ValueError("Quiz must contain at least one question")
    return draft


def draft_to_rows(
    draft: QuizDraft,
    *,
    chunk_indexes: list[int],
) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for index, question in enumerate(draft.questions):
        source_chunk_id = None
        if question.source_chunk_index is not None:
            chunk_position = min(question.source_chunk_index, len(chunk_indexes) - 1)
            source_chunk_id = str(chunk_indexes[chunk_position])
        rows.append(
            {
                "question_text": question.question_text,
                "options": question.options,
                "correct_option_index": question.correct_option_index,
                "explanation": question.explanation,
                "source_chunk_id": source_chunk_id,
                "sort_order": index,
            }
        )
    return rows
