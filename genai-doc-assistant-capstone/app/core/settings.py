"""
settings.py
-----------
Centralized application settings loader.

Responsibilities:
- Resolve project root reliably (monorepo-safe)
- Load config.yaml
- Load .env and apply dynamic env overrides
- Inject base_dir into config
- Produce a fully validated AppConfig instance
- Expose `settings` globally
"""

from pathlib import Path
import logging

from pycorekit.utils.config_loader import ConfigLoader
from app.schema import config_schema

logger = logging.getLogger(__name__)

CURRENT_FILE = Path(__file__).resolve()
PROJECT_DIR = CURRENT_FILE.parents[2]
CONFIG_PATH = PROJECT_DIR / "config.yaml"
ENV_PATH = PROJECT_DIR / ".env"

settings = ConfigLoader(
    CONFIG_PATH,
    base_dir=PROJECT_DIR,
    env_file=ENV_PATH,
    env_prefix="APP",
).load_typed(config_schema.AppConfig)

logger.debug(
    "Settings loaded",
    extra={
        "upload_dir": str(settings.paths.upload_dir),
        "vector_db_path": str(settings.paths.vector_db_path),
        "log_dir": str(settings.paths.log_dir),
    },
)
