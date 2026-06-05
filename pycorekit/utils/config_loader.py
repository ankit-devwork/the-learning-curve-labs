"""
Config Loader Utility

This module provides a unified way to load configuration files
(YAML-based) with optional:
- Environment overrides
- Typed Pydantic validation
- Caching
- Auto-reload support (optional)

Usage:
    from pycorekit.utils.config_loader import ConfigLoader

    config = ConfigLoader("configs/settings.yaml").load()
    print(config["database"]["host"])

Typed usage:
    class AppConfig(BaseModel):
        debug: bool
        database: dict

    cfg = ConfigLoader("configs/settings.yaml").load_typed(AppConfig)
"""

import os
import yaml
import time
from typing import Any, Dict, Optional, Type, TypeVar
from functools import lru_cache
from pydantic import BaseModel

from pycorekit.utils.yaml_loader import load_yaml


T = TypeVar("T", bound=BaseModel)


class ConfigLoader:
    """
    High-level configuration loader with:
    - YAML loading
    - Environment overrides
    - Typed Pydantic validation
    - Optional auto-reload
    """

    def __init__(self, path: str, auto_reload: bool = False):
        """
        Args:
            path (str): Path to YAML config file.
            auto_reload (bool): If True, reload file when modified.
        """
        self.path = path
        self.auto_reload = auto_reload
        self._last_mtime = None
        self._cached_config = None

    # ---------------------------------------------------------
    # INTERNAL HELPERS
    # ---------------------------------------------------------

    def _file_modified(self) -> bool:
        """Check if file changed since last load."""
        if not self.auto_reload:
            return False

        try:
            mtime = os.path.getmtime(self.path)
        except FileNotFoundError:
            return False

        if self._last_mtime is None:
            self._last_mtime = mtime
            return False

        if mtime != self._last_mtime:
            self._last_mtime = mtime
            return True

        return False

    def _apply_env_overrides(self, config: Dict[str, Any]) -> Dict[str, Any]:
        """
        Override config values using environment variables.

        Example:
            YAML:
                database:
                    host: localhost

            ENV:
                DATABASE_HOST=prod-db.example.com
        """
        flat_env = {k.lower(): v for k, v in os.environ.items()}

        def apply(prefix: str, node: Dict[str, Any]):
            for key, value in node.items():
                env_key = f"{prefix}_{key}".lower()
                if isinstance(value, dict):
                    apply(env_key, value)
                else:
                    if env_key in flat_env:
                        node[key] = flat_env[env_key]

        apply("", config)
        return config

    # ---------------------------------------------------------
    # PUBLIC API
    # ---------------------------------------------------------

    def load(self) -> Dict[str, Any]:
        """
        Load YAML config with optional auto-reload and env overrides.
        """
        if self._cached_config is None or self._file_modified():
            config = load_yaml(self.path)
            config = self._apply_env_overrides(config)
            self._cached_config = config

        return self._cached_config

    def load_typed(self, model: Type[T]) -> T:
        """
        Load YAML config and validate it using a Pydantic model.

        Args:
            model (BaseModel): Pydantic model class.

        Returns:
            model instance
        """
        data = self.load()
        return model(**data)
