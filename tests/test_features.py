import pandas as pd
import pytest

from src.data.features import build_item_feature_matrix, parse_category_tokens


def test_parse_category_tokens_removes_books_and_scopes_subcategories():
    raw = '["Books", "History", "Classic"]'
    tokens = parse_category_tokens(raw)
    assert tokens == ["History", "History > Classic"]


def test_parse_category_tokens_handles_multiple_paths():
    raw = '[["Books", "History", "Classic"], ["Books", "History", "Biography"]]'
    tokens = parse_category_tokens(raw)
    assert tokens == ["History", "History > Classic", "History > Biography"]


def test_build_item_feature_matrix_weights_and_scopes_categories():
    books = pd.DataFrame(
        [
            {
                "parent_asin": "A",
                "title": "History Classic",
                "categories": '["Books", "History", "Classic"]',
                "average_rating": 4.0,
            },
            {
                "parent_asin": "B",
                "title": "Children Classic",
                "categories": '["Books", "Children\'s Books", "Classic"]',
                "average_rating": 4.5,
            },
        ]
    )

    features, metadata = build_item_feature_matrix(
        books,
        feature_config={
            "numeric_columns": ["average_rating"],
            "category_top_k": 10,
            "author_top_k": 0,
        },
    )

    assert "Classic" not in metadata.category_vocab
    history_idx = metadata.category_vocab.index("History")
    history_classic_idx = metadata.category_vocab.index("History > Classic")
    children_idx = metadata.category_vocab.index("Children's Books")
    children_classic_idx = metadata.category_vocab.index("Children's Books > Classic")

    assert metadata.category_depths[history_idx] == 0
    assert metadata.category_depths[history_classic_idx] == 1

    # Book A
    assert features[0, history_idx] == 1.0
    assert features[0, history_classic_idx] == pytest.approx(0.5)
    assert features[0, children_idx] == 0.0
    assert features[0, children_classic_idx] == 0.0

    # Book B
    assert features[1, children_idx] == 1.0
    assert features[1, children_classic_idx] == pytest.approx(0.5)
    assert features[1, history_idx] == 0.0
    assert features[1, history_classic_idx] == 0.0
