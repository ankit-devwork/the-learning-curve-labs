import json
import re
from typing import Any

from pydantic import BaseModel, Field, field_validator


def normalize_concept_id(name: str) -> str:
    slug = re.sub(r"[^a-z0-9]+", "-", name.strip().lower())
    return slug.strip("-")[:120] or "concept"


class ConceptDraft(BaseModel):
    id: str | None = None
    name: str
    topic: str | None = None
    chunk_indexes: list[int] = Field(default_factory=list)

    @field_validator("name")
    @classmethod
    def strip_name(cls, value: str) -> str:
        cleaned = value.strip()
        if not cleaned:
            raise ValueError("Concept name is required")
        return cleaned

    def resolved_id(self) -> str:
        if self.id and self.id.strip():
            return normalize_concept_id(self.id)
        return normalize_concept_id(self.name)


class ConceptRelationshipDraft(BaseModel):
    source_id: str
    target_id: str
    type: str = Field(default="related_to")

    @field_validator("type")
    @classmethod
    def validate_type(cls, value: str) -> str:
        allowed = {"related_to", "prerequisite_for", "belongs_to"}
        normalized = value.strip().lower()
        if normalized not in allowed:
            raise ValueError(f"Invalid relationship type: {value}")
        return normalized

    @field_validator("source_id", "target_id")
    @classmethod
    def normalize_ids(cls, value: str) -> str:
        return normalize_concept_id(value)


class ConceptExtractionDraft(BaseModel):
    concepts: list[ConceptDraft] = Field(min_length=1)
    relationships: list[ConceptRelationshipDraft] = Field(default_factory=list)


def parse_concept_extraction(raw: str, *, max_concepts: int) -> ConceptExtractionDraft:
    text = raw.strip()
    if text.startswith("```"):
        text = text.split("\n", 1)[-1]
        if text.endswith("```"):
            text = text.rsplit("```", 1)[0]
    payload = json.loads(text)
    draft = ConceptExtractionDraft.model_validate(payload)
    draft.concepts = draft.concepts[:max_concepts]
    if not draft.concepts:
        raise ValueError("At least one concept is required")
    return draft


def draft_to_rows(
    draft: ConceptExtractionDraft,
    *,
    document_id: str,
) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
    concept_rows: list[dict[str, Any]] = []
    id_map: dict[str, str] = {}
    for concept in draft.concepts:
        concept_id = concept.resolved_id()
        id_map[concept.name.lower()] = concept_id
        id_map[concept_id] = concept_id
        concept_rows.append(
            {
                "document_id": document_id,
                "concept_id": concept_id,
                "name": concept.name,
                "topic": concept.topic,
                "chunk_indexes": sorted(set(concept.chunk_indexes)),
            }
        )

    relationship_rows: list[dict[str, Any]] = []
    for relationship in draft.relationships:
        source_id = normalize_concept_id(relationship.source_id)
        target_id = normalize_concept_id(relationship.target_id)
        relationship_rows.append(
            {
                "document_id": document_id,
                "source_concept_id": source_id,
                "target_concept_id": target_id,
                "relationship_type": relationship.type,
            }
        )
    return concept_rows, relationship_rows
