"""
typed_config.py

Typed configuration loader built on top of:
- YAML loader (safe, cached)
- Pydantic models (strict validation)

This allows you to define your configuration structure using Pydantic
and load YAML files directly into typed objects.

Example:
    class AppConfig(BaseModel):
        debug: bool
        database: dict

    config = load_typed_yaml("configs/app.yaml", AppConfig)
"""

from typing import Type, TypeVar
from pydantic import BaseModel, ValidationError

from pycorekit.utils.yaml_loader import load_yaml


T = TypeVar("T", bound=BaseModel)


def load_typed_yaml(path: str, model: Type[T]) -> T:
    """
    Load a YAML file and validate it using a Pydantic model.

    Args:
        path (str): Path to YAML file.
        model (BaseModel): Pydantic model class.

    Returns:
        model instance (T)

    Raises:
        ValidationError: If YAML content does not match the model.
        YamlLoadError: If YAML file cannot be loaded.
    """
    data = load_yaml(path)

    try:
        return model(**data)
    except ValidationError as e:
        raise ValidationError(
            f"Config validation failed for {path}: {e}"
        ) from e
