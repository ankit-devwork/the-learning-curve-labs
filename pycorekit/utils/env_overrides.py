"""
Apply environment-variable overrides on top of a YAML config dict.

Convention:
- Nested keys use double underscores: RAG__CHUNK_SIZE=500
- Optional prefix filters which vars apply: APP_RAG__CHUNK_SIZE (prefix="APP")
- Values are coerced using the existing YAML value type when present.
"""

from __future__ import annotations

import copy
import json
import os
import re
from typing import Any


def _get_nested(data: dict, parts: list[str]) -> Any:
    current: Any = data
    for part in parts:
        if not isinstance(current, dict) or part not in current:
            return None
        current = current[part]
    return current


def _set_nested(data: dict, parts: list[str], value: Any) -> None:
    current = data
    for part in parts[:-1]:
        if part not in current or not isinstance(current[part], dict):
            current[part] = {}
        current = current[part]
    current[parts[-1]] = value


def _normalize_key_part(part: str) -> str:
    return part.strip().lower()


def _parse_env_value(raw: str, hint: Any = None) -> Any:
    if hint is None:
        lowered = raw.strip().lower()
        if lowered in ("true", "false"):
            return lowered == "true"
        try:
            return int(raw)
        except ValueError:
            pass
        try:
            return float(raw)
        except ValueError:
            pass
        return raw

    if isinstance(hint, bool):
        return raw.strip().lower() in ("true", "1", "yes", "on")
    if isinstance(hint, int) and not isinstance(hint, bool):
        return int(raw)
    if isinstance(hint, float):
        return float(raw)
    if isinstance(hint, list):
        stripped = raw.strip()
        if stripped.startswith("["):
            try:
                parsed = json.loads(stripped)
                if isinstance(parsed, list):
                    return parsed
            except json.JSONDecodeError:
                pass
        return [item.strip() for item in raw.split(",") if item.strip()]
    return raw


def apply_env_overrides(data: dict, prefix: str | None = None) -> dict:
    """
    Deep-merge environment variables into a config dictionary.

    Args:
        data: Base config dict (typically from YAML).
        prefix: Optional prefix filter. With prefix="APP", only APP_* vars apply.

    Returns:
        A new dict with env overrides applied.

    Examples:
        RAG__CHUNK_SIZE=500           -> data["rag"]["chunk_size"] = 500
        MODELS__LLM_MODEL=groq/...    -> data["models"]["llm_model"] = "..."
        APP_ENV=prod (prefix="APP")   -> data["env"] = "prod"
    """
    result = copy.deepcopy(data)
    prefix_pattern = re.compile(r"^[A-Za-z0-9_]+$")

    if prefix and not prefix_pattern.match(prefix):
        raise ValueError(f"Invalid env prefix: {prefix}")

    for env_key, env_value in os.environ.items():
        key = env_key
        if prefix:
            prefix_token = f"{prefix}_"
            if not key.startswith(prefix_token):
                continue
            key = key[len(prefix_token) :]

        if "__" in key:
            parts = [_normalize_key_part(part) for part in key.split("__") if part]
            if not parts:
                continue
            hint = _get_nested(result, parts)
            parsed = _parse_env_value(env_value, hint)
            _set_nested(result, parts, parsed)
            continue

        normalized = _normalize_key_part(key)
        if normalized in result:
            hint = result[normalized]
            result[normalized] = _parse_env_value(env_value, hint)

    return result
