"""Study session plan helpers."""

from app.services.study_session_plan_utils import apply_learning_path_order


def test_apply_learning_path_order_reorders_document_steps():
    plan = {
        "workspace_id": "ws-1",
        "document_count": 3,
        "steps": [
            {"step": "focus", "label": "Focus", "duration_min": 2},
            {"step": "brief", "label": "Brief A", "document_id": "doc-a", "duration_min": 5},
            {"step": "flashcards", "label": "Cards A", "document_id": "doc-a", "duration_min": 5},
            {"step": "brief", "label": "Brief B", "document_id": "doc-b", "duration_min": 5},
            {"step": "flashcards", "label": "Cards B", "document_id": "doc-b", "duration_min": 5},
            {"step": "brief", "label": "Brief C", "document_id": "doc-c", "duration_min": 5},
            {"step": "set_quiz", "label": "Set quiz", "duration_min": 8},
        ],
        "estimated_minutes": 35,
    }
    path_nodes = [
        {"sort_order": 0, "document_id": "doc-c"},
        {"sort_order": 1, "document_id": "doc-a"},
        {"sort_order": 2, "document_id": "doc-b"},
    ]

    result = apply_learning_path_order(plan, path_nodes)

    ordered_doc_ids = [
        step["document_id"]
        for step in result["steps"]
        if step.get("step") in ("brief", "flashcards")
    ]
    assert ordered_doc_ids == ["doc-c", "doc-a", "doc-a", "doc-b", "doc-b"]
    assert result["steps"][0]["step"] == "focus"
    assert result["steps"][-1]["step"] == "set_quiz"
    assert result["path_document_order"] == ["doc-c", "doc-a", "doc-b"]
    assert result["estimated_minutes"] == 35


def test_apply_learning_path_order_noop_when_empty_path():
    plan = {"steps": [{"step": "brief", "document_id": "doc-a", "duration_min": 5}], "estimated_minutes": 5}
    assert apply_learning_path_order(plan, []) == plan


def test_apply_learning_path_order_appends_leftover_documents():
    plan = {
        "steps": [
            {"step": "brief", "document_id": "doc-a", "duration_min": 5},
            {"step": "brief", "document_id": "doc-z", "duration_min": 5},
        ],
        "estimated_minutes": 10,
    }
    path_nodes = [{"sort_order": 0, "document_id": "doc-a"}]

    result = apply_learning_path_order(plan, path_nodes)
    doc_ids = [step["document_id"] for step in result["steps"]]
    assert doc_ids == ["doc-a", "doc-z"]
