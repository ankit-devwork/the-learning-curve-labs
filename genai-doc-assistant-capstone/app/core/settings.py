"""
settings.py
-----------
Centralized application settings loader.

Responsibilities:
- Resolve project root reliably (monorepo-safe)
- Load config.yaml
- Inject base_dir into config
- Produce a fully validated AppConfig instance
- Expose `settings` globally
"""

from pathlib import Path
import logging

from pycorekit.utils.config_loader import ConfigLoader
from app.schema import config_schema

logger = logging.getLogger(__name__)

# ---------------------------------------------------------
# 1. Resolve project root RELIABLY
# ---------------------------------------------------------
CURRENT_FILE = Path(__file__).resolve()
PROJECT_DIR = CURRENT_FILE.parents[2]

CONFIG_PATH = PROJECT_DIR / "config.yaml"

print(f"PROJECT_DIR: {PROJECT_DIR}")
print(f"CONFIG_PATH: {CONFIG_PATH}")
print(f"CONFIG EXISTS: {CONFIG_PATH.exists()}")

# ---------------------------------------------------------
# 2. Load typed config with base_dir injection
# ---------------------------------------------------------
settings = ConfigLoader(CONFIG_PATH, base_dir=PROJECT_DIR).load_typed(
    config_schema.AppConfig
)

# ---------------------------------------------------------
# 3. Print resolved absolute paths
# ---------------------------------------------------------
print(f"UPLOAD_DIR (resolved): {settings.paths.upload_dir}")
print(f"VECTOR_DB_PATH (resolved): {settings.paths.vector_db_path}")
print(f"LOG_DIR (resolved): {settings.paths.log_dir}")
