"""Phase 14 study enhancement tests."""

from app.services.export_utils import slide_deck_to_markdown
from app.services.flashcard_srs_service import update_srs_after_review


class _FakeTable:
    def __init__(self, store: dict, name: str):
        self.store = store
        self.name = name
        self._filters: dict = {}

    def select(self, *_args, **_kwargs):
        return self

    def eq(self, key, value):
        self._filters[key] = value
        return self

    def limit(self, *_args, **_kwargs):
        return self

    def execute(self):
        if self.name == "flashcard_srs_states":
            rows = [
                row
                for row in self.store.get("flashcard_srs_states", [])
                if all(row.get(k) == v for k, v in self._filters.items())
            ]
            return type("Result", (), {"data": rows[:1]})()
        return type("Result", (), {"data": []})()

    def upsert(self, payload):
        rows = self.store.setdefault("flashcard_srs_states", [])
        rows = [row for row in rows if not (
            row.get("user_id") == payload.get("user_id")
            and row.get("flashcard_id") == payload.get("flashcard_id")
        )]
        rows.append(payload)
        self.store["flashcard_srs_states"] = rows
        return type("Result", (), {"data": [payload]})()


class _FakeClient:
    def __init__(self):
        self.store: dict = {}

    def table(self, name: str):
        return _FakeTable(self.store, name)


class _User:
    id = "user-1"


def test_update_srs_after_review_increases_interval_when_knew():
    client = _FakeClient()
    first = update_srs_after_review(client, _User(), flashcard_id="card-1", knew=True)
    second = update_srs_after_review(client, _User(), flashcard_id="card-1", knew=True)
    assert first["interval_days"] == 1
    assert second["interval_days"] >= 1
    assert second["repetitions"] == 2


def test_slide_deck_to_markdown_renders_slides():
    markdown = slide_deck_to_markdown(
        title="Biology",
        content={
            "slides": [
                {
                    "slide_number": 1,
                    "title": "Intro",
                    "bullets": ["Cell basics"],
                    "speaker_notes": "Keep it short",
                }
            ]
        },
    )
    assert "# Biology" in markdown
    assert "## Slide 1: Intro" in markdown
    assert "- Cell basics" in markdown
