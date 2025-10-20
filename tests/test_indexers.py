import pytest

from src.data.indexers import IndexMapping, build_index_mapping


def test_build_index_mapping_preserves_order():
    mapping = build_index_mapping(["a", "b", "a", "c"])

    assert isinstance(mapping, IndexMapping)
    assert mapping.index_to_id == ["a", "b", "c"]
    assert mapping.id_to_index == {"a": 0, "b": 1, "c": 2}
    assert mapping.to_index("b") == 1
    assert mapping.to_id(2) == "c"


def test_index_mapping_missing_id():
    mapping = build_index_mapping(["x"])

    with pytest.raises(KeyError):
        mapping.to_index("y")

