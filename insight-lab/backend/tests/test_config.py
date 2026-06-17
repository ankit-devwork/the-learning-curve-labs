from pathlib import Path

from app.core.config import BACKEND_DIR, ENV_PATH, settings


def test_env_path_points_to_backend_dotenv():
    assert BACKEND_DIR.name == "backend"
    assert ENV_PATH == BACKEND_DIR / ".env"
    assert ENV_PATH.parent == BACKEND_DIR


def test_settings_loads_from_backend_env_file():
    # Smoke test — Settings object initializes with backend/.env path configured.
    assert isinstance(settings.app_env, str)
