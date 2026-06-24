"""Tests for study sheet invite helpers."""

from app.services.upload import ensure_profile
from app.core.auth import AuthUser


class _FakeTable:
    def __init__(self):
        self.last_upsert = None

    def upsert(self, row, on_conflict=None):
        self.last_upsert = (row, on_conflict)
        return self

    def execute(self):
        return type("R", (), {"data": []})()


class _FakeClient:
    def __init__(self):
        self.table_obj = _FakeTable()

    def table(self, name):
        assert name == "profiles"
        return self.table_obj


def test_ensure_profile_syncs_email():
    client = _FakeClient()
    user = AuthUser(id="user-1", email="Student@School.edu")
    ensure_profile(client, user)
    assert client.table_obj.last_upsert is not None
    row, _on_conflict = client.table_obj.last_upsert
    assert row["id"] == "user-1"
    assert row["email"] == "student@school.edu"
