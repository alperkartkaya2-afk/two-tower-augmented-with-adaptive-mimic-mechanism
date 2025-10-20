"""
Indexing utilities that map raw identifiers to contiguous integer ranges.

Keeping these mappings in a dedicated module makes it simple to swap in hashed
indexers, sharded lookups, or persistence layers without touching the rest of
the pipeline.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Iterable


@dataclass(frozen=True)
class IndexMapping:
    """Bidirectional mapping between raw IDs and contiguous indices."""

    id_to_index: dict[str, int]
    index_to_id: list[str]

    def __len__(self) -> int:  # pragma: no cover - trivial
        return len(self.index_to_id)

    def to_index(self, raw_id: str) -> int:
        try:
            return self.id_to_index[raw_id]
        except KeyError as exc:  # pragma: no cover - defensive branch
            raise KeyError(f"ID '{raw_id}' missing from index mapping") from exc

    def to_id(self, index: int) -> str:
        try:
            return self.index_to_id[index]
        except IndexError as exc:  # pragma: no cover - defensive branch
            raise IndexError(f"Index {index} out of bounds for mapping") from exc


def build_index_mapping(values: Iterable[str]) -> IndexMapping:
    """
    Create an IndexMapping that preserves the order of first appearance.

    Parameters
    ----------
    values:
        Iterable of raw identifiers (user IDs, item ASINs, etc.).
    """
    id_to_index: dict[str, int] = {}
    index_to_id: list[str] = []

    for value in values:
        if value not in id_to_index:
            id_to_index[value] = len(index_to_id)
            index_to_id.append(value)

    return IndexMapping(id_to_index=id_to_index, index_to_id=index_to_id)

