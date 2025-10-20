"""Feature correlation diagnostics."""

from __future__ import annotations

from typing import Iterable, Sequence

import numpy as np
from scipy import stats


def compute_feature_correlations(
    feature_matrix: np.ndarray,
    scores: np.ndarray,
    feature_names: Sequence[str],
    *,
    top_k: int | None = None,
    min_variance: float = 1e-8,
) -> list[dict[str, float]]:
    """
    Compute Pearson correlation (and p-values) between features and model scores.

    Parameters
    ----------
    feature_matrix:
        2D array of shape (n_samples, n_features).
    scores:
        1D array of length n_samples containing model scores.
    feature_names:
        Names corresponding to feature columns.
    top_k:
        If provided, return only the top-k features by absolute correlation.
    min_variance:
        Minimum variance threshold to consider a feature (avoid constant columns).
    """
    if feature_matrix.size == 0 or feature_matrix.shape[0] < 3:
        return []

    correlations: list[dict[str, float]] = []
    scores = np.asarray(scores, dtype=np.float64)

    for idx, name in enumerate(feature_names):
        column = feature_matrix[:, idx].astype(np.float64)
        if np.var(column) < min_variance:
            continue
        try:
            r, p = stats.pearsonr(column, scores)
        except Exception:
            continue
        correlations.append(
            {
                "feature": name,
                "pearson_r": float(r),
                "p_value": float(p),
            }
        )

    correlations.sort(key=lambda x: abs(x["pearson_r"]), reverse=True)
    if top_k is not None and len(correlations) > top_k:
        correlations = correlations[: top_k]
    return correlations
