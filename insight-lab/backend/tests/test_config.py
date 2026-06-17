import os
from pathlib import Path

from app.core.config import BACKEND_DIR, ENV_PATH, _apply_env_file, config_diagnostics


def test_env_path_points_to_backend_dotenv():
    assert BACKEND_DIR.name == "backend"
    assert ENV_PATH == BACKEND_DIR / ".env"
    assert ENV_PATH.parent == BACKEND_DIR


def test_apply_env_file_fills_blank_os_env(tmp_path, monkeypatch):
    env_file = tmp_path / ".env"
    env_file.write_text("SUPABASE_URL=https://example.supabase.co\n", encoding="utf-8")
    monkeypatch.setenv("SUPABASE_URL", "")
    _apply_env_file(env_file)
    assert os.environ["SUPABASE_URL"] == "https://example.supabase.co"


def test_apply_env_file_does_not_override_nonempty_os_env(tmp_path, monkeypatch):
    env_file = tmp_path / ".env"
    env_file.write_text("SUPABASE_URL=https://from-file.supabase.co\n", encoding="utf-8")
    monkeypatch.setenv("SUPABASE_URL", "https://from-os.supabase.co")
    _apply_env_file(env_file)
    assert os.environ["SUPABASE_URL"] == "https://from-os.supabase.co"


def test_apply_env_file_handles_utf8_bom(tmp_path, monkeypatch):
    env_file = tmp_path / ".env"
    env_file.write_bytes(
        b"\xef\xbb\xbfSUPABASE_URL=https://bom-example.supabase.co\n",
    )
    monkeypatch.delenv("SUPABASE_URL", raising=False)
    _apply_env_file(env_file)
    assert os.environ["SUPABASE_URL"] == "https://bom-example.supabase.co"


def test_config_diagnostics_shape():
    diag = config_diagnostics()
    assert "env_file_exists" in diag
    assert "supabase_url_configured" in diag
