"""
Feature engineering utilities for the two-tower pipeline.

Transforms raw book/user metadata into structured tensors suitable for tower
encoders, including categorical statistics and lightweight text-derived signals.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Literal

import numpy as np
import pandas as pd
from loguru import logger

from .features import FeatureMetadata, build_item_feature_matrix, build_user_feature_matrix
from .indexers import IndexMapping, build_index_mapping
from .loaders import DatasetArtifacts


@dataclass(frozen=True)
class TrainingDataset:
    """
    Placeholder structure representing tensors or arrays required for training.

    The concrete implementation will likely host encoded user/item features,
    lookup indices, and potentially auxiliary labels for mimic objectives.
    """

    users: pd.DataFrame
    items: pd.DataFrame
    interactions: pd.DataFrame
    user_mapping: IndexMapping
    item_mapping: IndexMapping
    user_positive_items: dict[int, set[int]]
    item_feature_matrix: np.ndarray
    user_feature_matrix: np.ndarray
    feature_metadata: FeatureMetadata


def build_training_dataset(
    dataset: DatasetArtifacts,
    *,
    stage: Literal["train", "eval"] = "train",
    feature_config: dict | None = None,
    min_user_interactions: int = 0,
    min_item_interactions: int = 0,
) -> TrainingDataset:
    """
    Convert raw pandas frames into model-ready artefacts.

    Parameters
    ----------
    dataset:
        Raw books and interactions loaded from disk.
    stage:
        Allows callers to swap out sampling or augmentation behaviour for
        training vs. evaluation flows.
    """
    if stage not in {"train", "eval"}:
        raise ValueError("stage must be either 'train' or 'eval'")

    books = (
        dataset.books.dropna(subset=["parent_asin"])
        .drop_duplicates(subset=["parent_asin"])
        .copy()
    )
    books["parent_asin"] = books["parent_asin"].astype(str)

    interactions = dataset.interactions.dropna(subset=["parent_asin", "userId"]).copy()
    interactions["parent_asin"] = interactions["parent_asin"].astype(str)
    interactions["userId"] = interactions["userId"].astype(str)

    # Ensure interactions reference items we have metadata for.
    items_with_metadata = set(books["parent_asin"])
    interactions = interactions[
        interactions["parent_asin"].isin(items_with_metadata)
    ].reset_index(drop=True)

    min_user_interactions = max(int(min_user_interactions), 0)
    min_item_interactions = max(int(min_item_interactions), 0)

    if interactions.empty:
        logger.warning("No interactions remain after metadata alignment.")
    elif min_user_interactions > 0 or min_item_interactions > 0:
        before_filter = len(interactions)
        prev_size = -1
        while prev_size != len(interactions):
            prev_size = len(interactions)
            if min_item_interactions > 0 and not interactions.empty:
                item_counts = interactions["parent_asin"].value_counts()
                valid_items = item_counts[item_counts >= min_item_interactions].index
                interactions = interactions[interactions["parent_asin"].isin(valid_items)]
            if min_user_interactions > 0 and not interactions.empty:
                user_counts = interactions["userId"].value_counts()
                valid_users = user_counts[user_counts >= min_user_interactions].index
                interactions = interactions[interactions["userId"].isin(valid_users)]
            interactions = interactions.reset_index(drop=True)

        filtered = before_filter - len(interactions)
        if filtered > 0:
            logger.info(
                "Filtered {} interactions using min_user_interactions={} and min_item_interactions={}.",
                filtered,
                min_user_interactions,
                min_item_interactions,
            )
        if interactions.empty:
            logger.warning(
                "All interactions were filtered out by frequency thresholds (user>={}, item>={}).",
                min_user_interactions,
                min_item_interactions,
            )

    if not interactions.empty:
        items_with_usage = set(interactions["parent_asin"])
        books = books[books["parent_asin"].isin(items_with_usage)].reset_index(drop=True)

    item_mapping = build_index_mapping(books["parent_asin"])
    user_mapping = build_index_mapping(interactions["userId"])

    interactions["item_idx"] = interactions["parent_asin"].map(
        item_mapping.id_to_index
    )
    interactions["user_idx"] = interactions["userId"].map(user_mapping.id_to_index)
    interactions["item_idx"] = interactions["item_idx"].astype("int64")
    interactions["user_idx"] = interactions["user_idx"].astype("int64")

    users = pd.DataFrame(
        {
            "userId": user_mapping.index_to_id,
            "user_idx": range(len(user_mapping)),
        }
    ).astype({"user_idx": "int64"})

    books["item_idx"] = (
        books["parent_asin"].map(item_mapping.id_to_index).astype("int64")
    )

    item_feature_matrix, feature_metadata = build_item_feature_matrix(
        books, feature_config
    )
    user_feature_matrix = build_user_feature_matrix(
        interactions,
        item_feature_matrix,
        num_users=len(user_mapping),
        aggregation=str((feature_config or {}).get("user_aggregation", "mean")),
    )

    user_positive_items = {
        int(user_idx): set(map(int, group["item_idx"].tolist()))
        for user_idx, group in interactions.groupby("user_idx")
    }

    return TrainingDataset(
        users=users,
        items=books,
        interactions=interactions,
        user_mapping=user_mapping,
        item_mapping=item_mapping,
        user_positive_items=user_positive_items,
        item_feature_matrix=item_feature_matrix,
        user_feature_matrix=user_feature_matrix,
        feature_metadata=feature_metadata,
    )
