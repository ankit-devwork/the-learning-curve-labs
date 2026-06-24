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


def quiz_to_qti_xml(*, title: str, questions: list[dict]) -> str:
    """Minimal QTI 1.2 assessment XML for LMS import."""
    import html

    items: list[str] = []
    for index, question in enumerate(questions, start=1):
        options = question.get("options") or []
        correct = int(question.get("correct_option_index", 0))
        resp_labels = []
        for opt_index, option in enumerate(options):
            identifier = chr(ord("A") + opt_index)
            resp_labels.append(
                f'<response_label ident="{identifier}"><material><mattext texttype="text/plain">'
                f"{html.escape(str(option))}</mattext></material></response_label>"
            )
        correct_id = chr(ord("A") + correct) if options else "A"
        items.append(
            f"""
    <item ident="item{index}" title="Question {index}">
      <presentation>
        <material><mattext texttype="text/plain">{html.escape(str(question.get('question_text', '')))}</mattext></material>
        <response_lid ident="response{index}" rcardinality="Single">
          <render_choice>{''.join(resp_labels)}</render_choice>
        </response_lid>
      </presentation>
      <resprocessing>
        <respcondition continue="No">
          <conditionvar><varequal respident="response{index}">{correct_id}</varequal></conditionvar>
          <setvar action="Set" varname="SCORE">100</setvar>
        </respcondition>
      </resprocessing>
    </item>"""
        )

    safe_title = html.escape(title)
    return f"""<?xml version="1.0" encoding="UTF-8"?>
<questestinterop xmlns="http://www.imsglobal.org/xsd/ims_qtiasiv1p2">
  <assessment ident="insightlab-quiz" title="{safe_title}">
    <section>{''.join(items)}
    </section>
  </assessment>
</questestinterop>
"""


def course_pack_to_markdown(*, workspace_name: str, documents: list[dict]) -> str:
    parts = [f"# Course pack — {workspace_name}", ""]
    for doc in documents:
        filename = doc.get("filename", "Document")
        file_type = doc.get("file_type")
        heading = f"## {filename}"
        if file_type == "excel":
            heading = f"## {filename} (spreadsheet)"
        parts.append(heading)
        summary = doc.get("summary") or doc.get("artifacts", {}).get("summary")
        if summary:
            parts.append(str(summary))
            parts.append("")
    return "\n".join(parts).strip() + "\n"


def build_imsmanifest_xml(*, title: str, resources: list[dict[str, str]]) -> str:
    """Minimal IMS Common Cartridge 1.1 manifest for Canvas import."""
    import html

    safe_title = html.escape(title)
    org_items = []
    resource_blocks = []

    for index, resource in enumerate(resources, start=1):
        ident = resource.get("identifier") or f"res{index}"
        item_ident = f"item{index}"
        href = html.escape(resource.get("href") or "")
        res_title = html.escape(resource.get("title") or f"Resource {index}")
        res_type = html.escape(resource.get("type") or "webcontent")

        org_items.append(
            f'      <item identifier="{item_ident}" identifierref="{ident}">'
            f"<title>{res_title}</title></item>"
        )
        resource_blocks.append(
            f"""    <resource identifier="{ident}" type="{res_type}" href="{href}">
      <file href="{href}"/>
    </resource>"""
        )

    items_xml = "\n".join(org_items) if org_items else '      <item identifier="item1"><title>Course</title></item>'
    resources_xml = "\n".join(resource_blocks) if resource_blocks else ""

    return f"""<?xml version="1.0" encoding="UTF-8"?>
<manifest identifier="insightlab-{safe_title.replace(' ', '-')}"
  xmlns="http://www.imsglobal.org/xsd/imsccv1p1/imscp_v1p1"
  xmlns:lom="http://ltsc.ieee.org/xsd/imsccv1p1/LOM/resource"
  xmlns:xsi="http://www.w3.org/2001/XMLSchema-instance">
  <metadata>
    <schema>IMS Common Cartridge</schema>
    <schemaversion>1.1.0</schemaversion>
    <lom:lom>
      <lom:general><lom:title><lom:string>{safe_title}</lom:string></lom:title></lom:general>
    </lom:lom>
  </metadata>
  <organizations>
    <organization identifier="org" structure="rooted-hierarchy">
      <title>{safe_title}</title>
      <item identifier="root">
        <title>{safe_title}</title>
{items_xml}
      </item>
    </organization>
  </organizations>
  <resources>
{resources_xml}
  </resources>
</manifest>
"""

