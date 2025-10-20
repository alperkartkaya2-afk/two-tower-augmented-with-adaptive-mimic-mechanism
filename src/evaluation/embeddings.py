"""Embedding diagnostic utilities."""

from __future__ import annotations

import random
from typing import Iterable, Sequence

import numpy as np
import torch
from loguru import logger


def _parse_category_field(raw_value: str | Sequence[str] | float | None) -> list[str]:
    if raw_value is None or (isinstance(raw_value, float) and np.isnan(raw_value)):
        return []
    if isinstance(raw_value, list):
        return [str(v).strip() for v in raw_value if str(v).strip()]
    text = str(raw_value)
    if not text:
        return []
    cleaned = text.strip("[]")
    return [part.strip().strip("'\"") for part in cleaned.split(",") if part.strip()]


def summarize_embedding_norms(embeddings: torch.Tensor, *, label: str) -> dict[str, float]:
    norms = torch.norm(embeddings, dim=-1).cpu().numpy()
    summary = {
        "label": label,
        "count": len(norms),
        "mean": float(np.mean(norms)) if norms.size else 0.0,
        "std": float(np.std(norms)) if norms.size else 0.0,
        "min": float(np.min(norms)) if norms.size else 0.0,
        "max": float(np.max(norms)) if norms.size else 0.0,
        "median": float(np.median(norms)) if norms.size else 0.0,
    }
    return summary


def analyze_item_neighbors(
    item_embeddings: torch.Tensor,
    items_frame,
    *,
    k: int = 10,
    sample_size: int = 200,
) -> dict[str, float]:
    if item_embeddings.shape[0] == 0:
        return {
            "sampled_items": 0,
            "category_overlap_mean": 0.0,
            "category_overlap_std": 0.0,
            "k": k,
        }

    indices = list(range(item_embeddings.shape[0]))
    if len(indices) > sample_size:
        indices = random.sample(indices, sample_size)

    embeddings = torch.nn.functional.normalize(item_embeddings, dim=-1)
    overlap_scores: list[float] = []

    for idx in indices:
        reference_vec = embeddings[idx : idx + 1]
        similarities = torch.matmul(reference_vec, embeddings.T).squeeze(0)
        similarities[idx] = -float("inf")
        neighbor_indices = torch.topk(similarities, k=k).indices.cpu().tolist()

        base_categories = set(_parse_category_field(items_frame.iloc[idx]["categories"]))
        if not base_categories:
            continue

        overlaps = 0
        for neighbor_idx in neighbor_indices:
            neighbor_categories = set(
                _parse_category_field(items_frame.iloc[neighbor_idx]["categories"])
            )
            if base_categories & neighbor_categories:
                overlaps += 1
        overlap_scores.append(overlaps / max(k, 1))

    if not overlap_scores:
        return {
            "sampled_items": 0,
            "category_overlap_mean": 0.0,
            "category_overlap_std": 0.0,
            "k": k,
        }

    summary = {
        "sampled_items": len(overlap_scores),
        "category_overlap_mean": float(np.mean(overlap_scores)),
        "category_overlap_std": float(np.std(overlap_scores)),
        "k": k,
    }
    return summary


def summarize_user_alignment(
    user_embeddings: torch.Tensor,
    user_feature_matrix: np.ndarray,
) -> dict[str, float]:
    if user_embeddings.shape[0] == 0 or user_feature_matrix.size == 0:
        return {"aligned_users": 0, "cosine_mean": 0.0, "cosine_std": 0.0}

    feature_tensor = torch.from_numpy(user_feature_matrix).to(user_embeddings.device)
    # Project features to embedding dimension via least squares when dims mismatch.
    if feature_tensor.shape[1] != user_embeddings.shape[1]:
        try:
            result = torch.linalg.lstsq(
                torch.nn.functional.pad(feature_tensor, (0, 1)),
                user_embeddings,
            )
            coeffs = result.solution[: feature_tensor.shape[1], :]
            projected = feature_tensor @ coeffs
        except RuntimeError as exc:
            logger.warning(
                "Failed to align user features to embedding dimension: {}", exc
            )
            return {"aligned_users": 0, "cosine_mean": 0.0, "cosine_std": 0.0}
    else:
        projected = feature_tensor

    projected = torch.nn.functional.normalize(projected, dim=-1)
    user_norm = torch.nn.functional.normalize(user_embeddings, dim=-1)
    cosines = torch.sum(projected * user_norm, dim=-1).cpu().numpy()
    if cosines.size == 0:
        return {"aligned_users": 0, "cosine_mean": 0.0, "cosine_std": 0.0}
    return {
        "aligned_users": int(len(cosines)),
        "cosine_mean": float(np.mean(cosines)),
        "cosine_std": float(np.std(cosines)),
    }
