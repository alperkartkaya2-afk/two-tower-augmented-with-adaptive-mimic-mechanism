"""
Typed data loading helpers for the two-tower scaffold.

Loaders default to the full `books.csv` / `users.csv` artifacts while gracefully
falling back to the trimmed samples when those large files are unavailable.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Optional

import pandas as pd
from loguru import logger


DEFAULT_BOOKS_FILENAME = "books.csv"
DEFAULT_INTERACTIONS_FILENAME = "users.csv"
SAMPLE_BOOKS_FILENAME = "books_trimmed.csv"
SAMPLE_INTERACTIONS_FILENAME = "users_trimmed.csv"


@dataclass(frozen=True)
class DatasetArtifacts:
    """Container for the raw datasets loaded from disk."""

    books: pd.DataFrame
    interactions: pd.DataFrame


def _read_csv(
    path: Path, *, dtype: Optional[dict[str, str]] = None, nrows: Optional[int] = None
) -> pd.DataFrame:
    if not path.exists():
        raise FileNotFoundError(f"Expected CSV at {path} but file was not found.")
    return pd.read_csv(path, dtype=dtype, nrows=nrows)


def load_books(
    data_dir: Path, *, filename: str | None = None, limit: Optional[int] = None
) -> pd.DataFrame:
    """
    Load and return the books metadata.

    Parameters
    ----------
    data_dir:
        Directory containing the target books CSV (default: `books.csv`).
    """
    target = filename or DEFAULT_BOOKS_FILENAME
    try:
        return _read_csv(data_dir / target, nrows=limit)
    except FileNotFoundError as exc:
        if filename is None and (data_dir / SAMPLE_BOOKS_FILENAME).exists():
            return _read_csv(data_dir / SAMPLE_BOOKS_FILENAME, nrows=limit)
        raise exc


def load_interactions(
    data_dir: Path,
    *,
    filename: str | None = None,
    limit: Optional[int] = None,
) -> pd.DataFrame:
    """
    Load and return the user-item interaction records.

    Parameters
    ----------
    data_dir:
        Directory containing the target interactions CSV (default: `users.csv`).
    """
    target = filename or DEFAULT_INTERACTIONS_FILENAME
    dtype = {"parent_asin": "string", "userId": "string", "timestamp": "Int64"}
    try:
        return _read_csv(data_dir / target, dtype=dtype, nrows=limit)
    except FileNotFoundError as exc:
        if filename is None and (data_dir / SAMPLE_INTERACTIONS_FILENAME).exists():
            return _read_csv(
                data_dir / SAMPLE_INTERACTIONS_FILENAME, dtype=dtype, nrows=limit
            )
        raise exc


def load_dataset(
    data_dir: Path,
    *,
    books_file: str | None = None,
    interactions_file: str | None = None,
    books_limit: Optional[int] = None,
    interactions_limit: Optional[int] = None,
) -> DatasetArtifacts:
    """
    Convenience wrapper that loads both books and interactions.

    Keeping dataset access centralized simplifies caching, schema validation,
    and future instrumentation (e.g. telemetry, sampling).
    """
    books = load_books(data_dir, filename=books_file, limit=books_limit)
    interactions = load_interactions(
        data_dir, filename=interactions_file, limit=interactions_limit
    )

    if not books.empty and "parent_asin" in books and "parent_asin" in interactions:
        valid_asins = set(books["parent_asin"].astype(str))
        before = len(interactions)
        interactions = interactions[
            interactions["parent_asin"].astype(str).isin(valid_asins)
        ].reset_index(drop=True)
        dropped = before - len(interactions)
        if dropped > 0:
            logger.info(
                "Filtered {} interaction rows referencing ASINs outside the books subset.",
                dropped,
            )

    return DatasetArtifacts(books=books, interactions=interactions)
