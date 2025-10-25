"""Configuration loading and manipulation helpers."""

from __future__ import annotations

import copy
from pathlib import Path
from typing import Any, Mapping, MutableMapping, Sequence

import yaml


def load_config(config_path: Path) -> Mapping[str, Any]:
    """
    Parse a YAML configuration file into a nested mapping.

    The function intentionally returns a generic mapping so callers can plug the
    result into dataclass factories, Pydantic models, or lightweight dict-based
    access patterns.
    """
    if not config_path.exists():
        raise FileNotFoundError(f"Configuration file not found: {config_path}")

    with config_path.open("r", encoding="utf-8") as handle:
        return yaml.safe_load(handle) or {}


def clone_config(config: Mapping[str, Any]) -> dict[str, Any]:
    """Return a deep copy of the configuration mapping."""
    return copy.deepcopy(config)


def set_by_dotted_path(
    config: MutableMapping[str, Any],
    dotted_key: str,
    value: Any,
) -> None:
    """
    Assign a value inside a nested mapping using dotted-path syntax.

    Examples
    --------
    >>> cfg = {"training": {"learning_rate": 0.001}}
    >>> set_by_dotted_path(cfg, "training.learning_rate", 0.01)
    >>> cfg["training"]["learning_rate"]
    0.01
    """
    keys: Sequence[str] = dotted_key.split(".")
    current: MutableMapping[str, Any] = config
    for key in keys[:-1]:
        if key not in current or not isinstance(current[key], MutableMapping):
            current[key] = {}
        current = current[key]  # type: ignore[assignment]
    current[keys[-1]] = value


def get_by_dotted_path(config: Mapping[str, Any], dotted_key: str, default: Any = None) -> Any:
    """Fetch a value from a nested mapping using dotted-path syntax."""
    current: Any = config
    for key in dotted_key.split("."):
        if not isinstance(current, Mapping) or key not in current:
            return default
        current = current[key]
    return current
