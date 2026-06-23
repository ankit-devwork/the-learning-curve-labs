"""Client-side export helpers mirrored for tests and optional server export."""


def chart_rows_to_csv(*, title: str, labels: list[str], values: list[float | int]) -> str:
    lines = ["label,value"]
    for label, value in zip(labels, values, strict=False):
        safe_label = str(label).replace('"', '""')
        if "," in safe_label or "\n" in safe_label:
            safe_label = f'"{safe_label}"'
        lines.append(f"{safe_label},{value}")
    _ = title
    return "\n".join(lines)


def flashcards_to_anki_csv(cards: list[dict], *, tag: str = "InsightLab") -> str:
    lines = ["Front,Back,Tags"]
    for card in cards:
        front = str(card.get("front", "")).replace('"', '""')
        back = str(card.get("back", "")).replace('"', '""')
        lines.append(f'"{front}","{back}","{tag}"')
    return "\n".join(lines)


def study_guide_to_markdown(*, title: str, content: dict) -> str:
    parts = [f"# {title}", ""]
    overview = content.get("overview")
    if overview:
        parts.extend(["## Overview", str(overview), ""])
    key_terms = content.get("key_terms") or []
    if key_terms:
        parts.append("## Key terms")
        for item in key_terms:
            parts.append(f"- **{item.get('term', '')}**: {item.get('definition', '')}")
        parts.append("")
    for section in content.get("sections") or []:
        parts.append(f"## {section.get('heading', 'Section')}")
        for bullet in section.get("bullets") or []:
            parts.append(f"- {bullet}")
        parts.append("")
    sample_questions = content.get("sample_questions") or []
    if sample_questions:
        parts.append("## Sample questions")
        for question in sample_questions:
            parts.append(f"- {question}")
    return "\n".join(parts).strip() + "\n"

