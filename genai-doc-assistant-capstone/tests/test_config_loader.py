from pathlib import Path

from pycorekit.utils.config_loader import ConfigLoader
from app.schema.config_schema import AppConfig

PROJECT_DIR = Path(__file__).resolve().parents[1]


def test_load_typed_config():
    settings = ConfigLoader(
        PROJECT_DIR / "config.yaml",
        base_dir=PROJECT_DIR,
        env_prefix="APP",
    ).load_typed(AppConfig)

    assert settings.models.embedding_dim == 768
    assert settings.file_upload.max_file_size_mb == 10
    assert settings.cache.enabled is True
