import pandas as pd

from src.data.loaders import DatasetArtifacts
from src.data.preprocessing import build_training_dataset


def test_build_training_dataset_creates_indices():
    books = pd.DataFrame(
        [
            {"parent_asin": "A", "title": "Book A"},
            {"parent_asin": "B", "title": "Book B"},
        ]
    )
    interactions = pd.DataFrame(
        [
            {"parent_asin": "A", "userId": "U1", "timestamp": 1},
            {"parent_asin": "B", "userId": "U2", "timestamp": 2},
        ]
    )

    dataset = DatasetArtifacts(books=books, interactions=interactions)
    training_dataset = build_training_dataset(
        dataset,
        stage="train",
        feature_config={
            "numeric_columns": ["timestamp"],
            "category_top_k": 2,
            "author_top_k": 2,
        },
    )

    assert len(training_dataset.user_mapping) == 2
    assert len(training_dataset.item_mapping) == 2
    assert set(training_dataset.interactions.columns).issuperset({"user_idx", "item_idx"})
    assert training_dataset.interactions["user_idx"].tolist() == [0, 1]
    assert training_dataset.interactions["item_idx"].tolist() == [0, 1]
    assert training_dataset.users["user_idx"].tolist() == [0, 1]
    assert training_dataset.items["item_idx"].tolist() == [0, 1]
    assert training_dataset.user_positive_items == {0: {0}, 1: {1}}
    assert training_dataset.item_feature_matrix.shape == (2, training_dataset.item_feature_matrix.shape[1])
    assert training_dataset.user_feature_matrix.shape == (2, training_dataset.user_feature_matrix.shape[1])
    assert training_dataset.item_feature_matrix.shape[1] >= 1
    assert training_dataset.feature_metadata.feature_dim == training_dataset.item_feature_matrix.shape[1]


def test_build_training_dataset_filters_low_frequency_entities():
    books = pd.DataFrame(
        [
            {"parent_asin": "A", "title": "Book A"},
            {"parent_asin": "B", "title": "Book B"},
        ]
    )
    interactions = pd.DataFrame(
        [
            {"parent_asin": "A", "userId": "U1", "timestamp": 1},
            {"parent_asin": "A", "userId": "U1", "timestamp": 2},
            {"parent_asin": "B", "userId": "U1", "timestamp": 3},
            {"parent_asin": "A", "userId": "U2", "timestamp": 4},
        ]
    )

    dataset = DatasetArtifacts(books=books, interactions=interactions)
    training_dataset = build_training_dataset(
        dataset,
        stage="train",
        min_user_interactions=2,
        min_item_interactions=2,
        feature_config={
            "numeric_columns": ["timestamp"],
            "category_top_k": 2,
            "author_top_k": 2,
        },
    )

    # Only user U1 and item A should remain with two interactions.
    assert len(training_dataset.user_mapping) == 1
    assert len(training_dataset.item_mapping) == 1
    assert training_dataset.interactions["user_idx"].nunique() == 1
    assert training_dataset.interactions["item_idx"].nunique() == 1
    assert set(training_dataset.users["userId"]) == {"U1"}
    assert set(training_dataset.items["parent_asin"]) == {"A"}
