import json
from typing import Any
from uuid import uuid4

from pydantic import BaseModel, Field

from app.services.concept_extraction import normalize_concept_id


class FlashcardDraft(BaseModel):
    front: str
    back: str
    source_chunk_index: int | None = Field(default=None, ge=0)


class FlashcardSetDraft(BaseModel):
    title: str
    cards: list[FlashcardDraft] = Field(min_length=1)


class StudyGuideSection(BaseModel):
    heading: str
    bullets: list[str] = Field(default_factory=list)


class StudyGuideKeyTerm(BaseModel):
    term: str
    definition: str = ""


class StudyGuideDraft(BaseModel):
    title: str
    overview: str
    key_terms: list[StudyGuideKeyTerm] = Field(default_factory=list)
    sections: list[StudyGuideSection] = Field(default_factory=list)
    sample_questions: list[str] = Field(default_factory=list)


class InfographicStatBlock(BaseModel):
    type: str = "stat"
    label: str
    value: str
    caption: str = ""


class InfographicBulletsBlock(BaseModel):
    type: str = "bullets"
    heading: str
    items: list[str] = Field(default_factory=list)


class InfographicComparisonBlock(BaseModel):
    type: str = "comparison"
    heading: str
    left_title: str
    left_items: list[str] = Field(default_factory=list)
    right_title: str
    right_items: list[str] = Field(default_factory=list)


class InfographicQuoteBlock(BaseModel):
    type: str = "quote"
    text: str
    attribution: str = ""


class InfographicDraft(BaseModel):
    title: str
    subtitle: str = ""
    theme: str = "blue"
    blocks: list[dict[str, Any]] = Field(default_factory=list)


def _strip_json_fence(raw: str) -> str:
    text = raw.strip()
    if text.startswith("```"):
        text = text.split("\n", 1)[-1]
        if text.endswith("```"):
            text = text.rsplit("```", 1)[0]
    return text.strip()


def parse_flashcard_draft(raw: str, *, max_cards: int) -> FlashcardSetDraft:
    payload = json.loads(_strip_json_fence(raw))
    draft = FlashcardSetDraft.model_validate(payload)
    draft.cards = draft.cards[:max_cards]
    if not draft.cards:
        raise ValueError("Flashcard set must contain at least one card")
    return draft


def parse_study_guide_draft(raw: str) -> StudyGuideDraft:
    payload = json.loads(_strip_json_fence(raw))
    return StudyGuideDraft.model_validate(payload)


def parse_infographic_draft(raw: str) -> InfographicDraft:
    payload = json.loads(_strip_json_fence(raw))
    draft = InfographicDraft.model_validate(payload)
    if not draft.blocks:
        raise ValueError("Infographic must contain at least one block")
    return draft


def flashcard_draft_to_rows(draft: FlashcardSetDraft, *, chunk_indexes: list[int]) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for index, card in enumerate(draft.cards):
        source_chunk_index = None
        if card.source_chunk_index is not None and chunk_indexes:
            position = min(card.source_chunk_index, len(chunk_indexes) - 1)
            source_chunk_index = chunk_indexes[position]
        rows.append(
            {
                "front": card.front.strip(),
                "back": card.back.strip(),
                "sort_order": index,
                "source_chunk_index": source_chunk_index,
            }
        )
    return rows


def study_guide_to_content(draft: StudyGuideDraft) -> dict[str, Any]:
    return {
        "title": draft.title.strip(),
        "overview": draft.overview.strip(),
        "key_terms": [
            {"term": item.term.strip(), "definition": item.definition.strip()}
            for item in draft.key_terms
            if item.term.strip()
        ],
        "sections": [
            {
                "heading": section.heading.strip(),
                "bullets": [bullet.strip() for bullet in section.bullets if bullet.strip()],
            }
            for section in draft.sections
            if section.heading.strip()
        ],
        "sample_questions": [question.strip() for question in draft.sample_questions if question.strip()],
    }


def normalize_study_term_slug(term: str) -> str:
    return normalize_concept_id(term) or "term"


def _normalize_infographic_block(block: dict[str, Any]) -> dict[str, Any] | None:
    block_type = str(block.get("type", "")).strip().lower()
    if block_type == "stat":
        label = str(block.get("label", "")).strip()
        value = str(block.get("value", "")).strip()
        if not label or not value:
            return None
        return {
            "type": "stat",
            "label": label,
            "value": value,
            "caption": str(block.get("caption", "")).strip(),
        }
    if block_type == "bullets":
        heading = str(block.get("heading", "")).strip()
        items = [str(item).strip() for item in block.get("items") or [] if str(item).strip()]
        if not heading or not items:
            return None
        return {"type": "bullets", "heading": heading, "items": items[:6]}
    if block_type == "comparison":
        heading = str(block.get("heading", "")).strip()
        left_title = str(block.get("left_title", "")).strip()
        right_title = str(block.get("right_title", "")).strip()
        left_items = [str(item).strip() for item in block.get("left_items") or [] if str(item).strip()]
        right_items = [str(item).strip() for item in block.get("right_items") or [] if str(item).strip()]
        if not heading or not left_title or not right_title:
            return None
        return {
            "type": "comparison",
            "heading": heading,
            "left_title": left_title,
            "left_items": left_items[:5],
            "right_title": right_title,
            "right_items": right_items[:5],
        }
    if block_type == "quote":
        text = str(block.get("text", "")).strip()
        if not text:
            return None
        return {
            "type": "quote",
            "text": text,
            "attribution": str(block.get("attribution", "")).strip(),
        }
    return None


def infographic_to_content(draft: InfographicDraft) -> dict[str, Any]:
    theme = draft.theme.strip().lower()
    if theme not in {"blue", "violet", "emerald", "amber", "rose", "cyan"}:
        theme = "blue"
    blocks: list[dict[str, Any]] = []
    for block in draft.blocks:
        if not isinstance(block, dict):
            continue
        normalized = _normalize_infographic_block(block)
        if normalized:
            blocks.append(normalized)
    if not blocks:
        raise ValueError("Infographic must contain at least one valid block")
    return {
        "title": draft.title.strip(),
        "subtitle": draft.subtitle.strip(),
        "theme": theme,
        "blocks": blocks[:8],
    }
