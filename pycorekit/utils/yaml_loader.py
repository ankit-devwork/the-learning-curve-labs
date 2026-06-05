"""
YAML Configuration Loader Utility

This module provides:
- Safe YAML loading
- Environment-aware overrides
- Automatic file existence validation
- Optional caching to avoid re-reading files
- A clean, reusable API for all projects

Usage:
    from pycorekit.utils.yaml_loader import load_yaml

    config = load_yaml("configs/settings.yaml")
"""

import os
import yaml
from functools import lru_cache
from typing import Any, Dict


class YamlLoadError(Exception):
    """Raised when a YAML file cannot be loaded or parsed."""
    pass


def _validate_path(path: str):
    """Ensure the YAML file exists before loading."""
    if not os.path.exists(path):
        raise YamlLoadError(f"YAML file not found: {path}")
    if not path.endswith((".yaml", ".yml")):
        raise YamlLoadError(f"Invalid YAML file extension: {path}")


@lru_cache(maxsize=32)
def load_yaml(path: str) -> Dict[str, Any]:
    """
    Load a YAML file safely with caching.

    - Validates file existence
    - Uses yaml.safe_load for security
    - Caches results to avoid repeated disk reads
    - Returns a Python dict

    Args:
        path (str): Path to the YAML file.

    Returns:
        dict: Parsed YAML content.

    Raises:
        YamlLoadError: If file missing or invalid YAML.
    """
    _validate_path(path)

    try:
        with open(path, "r", encoding="utf-8") as f:
            return yaml.safe_load(f) or {}
    except Exception as e:
        raise YamlLoadError(f"Failed to load YAML file '{path}': {e}")
