"""
Feature engineering utilities for user/item metadata.

These helpers focus on lightweight numerical, categorical, and text-derived
statistics that are cheap to compute yet expressive enough to bootstrap richer
encoders.
"""

from __future__ import annotations

import ast
from collections import Counter
from dataclasses import dataclass
from typing import Iterable, Sequence

import numpy as np
import pandas as pd


def _default_feature_config(config: dict | None) -> dict:
    cfg = config.copy() if config else {}
    cfg.setdefault(
        "numeric_columns",
        ["average_rating", "price", "rating_number"],
    )
    cfg.setdefault("category_top_k", 500)
    cfg.setdefault("author_top_k", 500)
    cfg.setdefault("user_aggregation", "mean")
    cfg.setdefault("text_features", {"title": True})
    return cfg


@dataclass(frozen=True)
class FeatureMetadata:
    """Describes the engineered feature space for reproducibility."""

    numeric_columns: list[str]
    numeric_mean: list[float]
    numeric_std: list[float]
    text_columns: list[str]
    text_mean: list[float]
    text_std: list[float]
    category_vocab: list[str]
    author_vocab: list[str]
    feature_dim: int

    def feature_names(self) -> list[str]:
        """Return feature names in the order they appear in item/user matrices."""
        names: list[str] = []
        names.extend(f"numeric:{col}" for col in self.numeric_columns)
        names.extend(f"text:{col}" for col in self.text_columns)
        names.extend(f"category:{cat}" for cat in self.category_vocab)
        names.extend(f"author:{author}" for author in self.author_vocab)
        return names


def _parse_categories(raw_value: str | float | None) -> list[str]:
    if pd.isna(raw_value):
        return []
    if isinstance(raw_value, list):
        return [str(item).strip() for item in raw_value if str(item).strip()]
    if isinstance(raw_value, str):
        text = raw_value.strip()
        if not text:
            return []
        try:
            parsed = ast.literal_eval(text)
            if isinstance(parsed, list):
                return [str(item).strip() for item in parsed if str(item).strip()]
        except (ValueError, SyntaxError):
            pass
        return [part.strip() for part in text.split(",") if part.strip()]
    return []


def _normalise_numeric(matrix: np.ndarray) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    mean = np.nanmean(matrix, axis=0)
    std = np.nanstd(matrix, axis=0)
    std = np.where(std == 0, 1.0, std)
    matrix = np.where(np.isnan(matrix), mean, matrix)
    normalised = (matrix - mean) / std
    return normalised.astype(np.float32), mean.astype(float), std.astype(float)


def _build_category_matrix(
    categories: Sequence[list[str]], *, top_k: int
) -> tuple[np.ndarray, list[str]]:
    counter: Counter[str] = Counter()
    for values in categories:
        counter.update(values)
    vocab = [cat for cat, _ in counter.most_common(top_k) if cat]
    if not vocab:
        return np.zeros((len(categories), 0), dtype=np.float32), []
    index = {cat: idx for idx, cat in enumerate(vocab)}
    matrix = np.zeros((len(categories), len(vocab)), dtype=np.float32)
    for row, values in enumerate(categories):
        for cat in values:
            idx = index.get(cat)
            if idx is not None:
                matrix[row, idx] = 1.0
    return matrix, vocab


def _build_author_matrix(authors: Sequence[str], *, top_k: int) -> tuple[np.ndarray, list[str]]:
    series = pd.Series(authors).fillna("Unknown").astype(str)
    counts = series.value_counts()
    vocab = list(counts.head(top_k).index)
    if not vocab:
        return np.zeros((len(series), 0), dtype=np.float32), []
    index = {author: idx for idx, author in enumerate(vocab)}
    matrix = np.zeros((len(series), len(vocab)), dtype=np.float32)
    for row, author in enumerate(series.tolist()):
        idx = index.get(author)
        if idx is not None:
            matrix[row, idx] = 1.0
    return matrix, vocab


def _compute_text_stats(titles: Iterable[str]) -> tuple[np.ndarray, list[str], list[float], list[float]]:
    words = []
    chars = []
    for title in titles:
        text = "" if pd.isna(title) else str(title)
        words.append(len(text.split()))
        chars.append(len(text))
    matrix = np.stack([words, chars], axis=1).astype(np.float32)
    normalised, mean, std = _normalise_numeric(matrix)
    return normalised, ["title_word_count", "title_char_count"], mean.tolist(), std.tolist()


def build_item_feature_matrix(
    books: pd.DataFrame,
    feature_config: dict | None = None,
) -> tuple[np.ndarray, FeatureMetadata]:
    """
    Generate item-side feature matrix enriched with numeric, categorical, and text statistics.

    Returns
    -------
    features:
        Float32 matrix with shape (num_items, feature_dim).
    metadata:
        Describes the feature ordering and normalisation statistics.
    """
    cfg = _default_feature_config(feature_config)

    available_numeric = [col for col in cfg.get("numeric_columns", []) if col in books]
    numeric_values = np.zeros((len(books), len(available_numeric)), dtype=np.float32)
    if available_numeric:
        numeric_frame = books[available_numeric].apply(pd.to_numeric, errors="coerce")
        numeric_values = numeric_frame.to_numpy(dtype=np.float32, copy=True)
        numeric_values, num_mean, num_std = _normalise_numeric(numeric_values)
    else:
        num_mean, num_std = [], []

    title_source = books["title"] if "title" in books else pd.Series([""] * len(books))
    title_stats, text_columns, text_mean, text_std = _compute_text_stats(title_source)

    if "categories" in books:
        raw_categories = books["categories"]
    else:
        raw_categories = pd.Series([[] for _ in range(len(books))])
    category_lists = raw_categories.apply(_parse_categories)
    category_matrix, category_vocab = _build_category_matrix(
        category_lists.tolist(), top_k=int(cfg.get("category_top_k", 500))
    )

    author_source = (
        books["author"]
        if "author" in books
        else pd.Series(["Unknown"] * len(books))
    )
    author_matrix, author_vocab = _build_author_matrix(
        author_source.tolist(),
        top_k=int(cfg.get("author_top_k", 500)),
    )

    feature_parts = [
        numeric_values,
        title_stats,
        category_matrix,
        author_matrix,
    ]
    features = (
        np.concatenate([part for part in feature_parts if part.size > 0], axis=1)
        if feature_parts
        else np.zeros((len(books), 0), dtype=np.float32)
    )

    metadata = FeatureMetadata(
        numeric_columns=available_numeric,
        numeric_mean=num_mean,
        numeric_std=num_std,
        text_columns=text_columns,
        text_mean=text_mean,
        text_std=text_std,
        category_vocab=category_vocab,
        author_vocab=author_vocab,
        feature_dim=int(features.shape[1]),
    )
    return features.astype(np.float32, copy=False), metadata


def build_user_feature_matrix(
    interactions: pd.DataFrame,
    item_features: np.ndarray,
    *,
    num_users: int,
    aggregation: str = "mean",
) -> np.ndarray:
    """
    Aggregate item features to create user-side representations.

    Parameters
    ----------
    interactions:
        Frame containing at least `user_idx` and `item_idx` columns.
    item_features:
        Item feature matrix aligned with item indices.
    num_users:
        Total number of distinct user indices.
    aggregation:
        Strategy for pooling item features: 'mean', 'sum', or 'max'.
    """
    if item_features.size == 0:
        return np.zeros((num_users, 0), dtype=np.float32)

    agg = aggregation.lower()
    if agg not in {"mean", "sum", "max"}:
        raise ValueError("aggregation must be one of {'mean', 'sum', 'max'}")

    user_features = np.zeros((num_users, item_features.shape[1]), dtype=np.float32)

    if interactions.empty:
        return user_features

    for user_idx, group in interactions.groupby("user_idx"):
        item_indices = group["item_idx"].to_numpy(dtype=int, copy=False)
        if item_indices.size == 0:
            continue
        selected = item_features[item_indices]
        if agg == "mean":
            pooled = selected.mean(axis=0)
        elif agg == "sum":
            pooled = selected.sum(axis=0)
        else:  # max
            pooled = selected.max(axis=0)
        user_features[int(user_idx)] = pooled.astype(np.float32, copy=False)

    return user_features
