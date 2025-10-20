"""Configuration loading helpers."""

from __future__ import annotations

from pathlib import Path
from typing import Any, Mapping

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

