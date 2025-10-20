"""Evaluation helpers for ranking quality, embedding diagnostics, and reporting."""

from .metrics import compute_ranking_metrics, per_user_metrics  # noqa: F401
from .embeddings import (
    analyze_item_neighbors,
    summarize_embedding_norms,
    summarize_user_alignment,
)  # noqa: F401
from .feature_correlation import compute_feature_correlations  # noqa: F401
