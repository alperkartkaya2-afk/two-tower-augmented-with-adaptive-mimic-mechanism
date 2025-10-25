"""Ranking metric utilities."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Iterable, Sequence

import numpy as np


@dataclass(frozen=True)
class RankingMetrics:
    recall: dict[int, float]
    precision: dict[int, float]
    ndcg: dict[int, float]
    hit_rate: dict[int, float]
    map: dict[int, float]
    mrr: float
    per_user: list[dict[str, float]]


def _dcg(relevance: Sequence[int]) -> float:
    return sum(
        rel / np.log2(idx + 2) for idx, rel in enumerate(relevance)
    )


def _ndcg_at_k(predicted: Sequence[int], ground_truth: set[int], k: int) -> float:
    relevance = [1 if item in ground_truth else 0 for item in predicted[:k]]
    dcg = _dcg(relevance)
    ideal = _dcg([1] * min(k, len(ground_truth)))
    if ideal == 0:
        return 0.0
    return dcg / ideal


def _average_precision(predicted: Sequence[int], ground_truth: set[int], k: int) -> float:
    hits = 0
    sum_precision = 0.0
    for idx, item in enumerate(predicted[:k], start=1):
        if item in ground_truth:
            hits += 1
            sum_precision += hits / idx
    if not ground_truth:
        return 0.0
    return sum_precision / min(len(ground_truth), k)


def per_user_metrics(
    predicted: Sequence[int],
    ground_truth: set[int],
    k_values: Iterable[int],
) -> dict[str, float]:
    metrics: dict[str, float] = {}
    k_sorted = sorted(k_values)
    max_k = max(k_sorted) if k_sorted else len(predicted)
    for k in k_sorted:
        topk = predicted[:k]
        hits = len(set(topk) & ground_truth)
        metrics[f"recall@{k}"] = hits / max(len(ground_truth), 1)
        metrics[f"precision@{k}"] = hits / max(k, 1)
        metrics[f"hit_rate@{k}"] = 1.0 if hits > 0 else 0.0
        metrics[f"ndcg@{k}"] = _ndcg_at_k(predicted, ground_truth, k)
        metrics[f"map@{k}"] = _average_precision(predicted, ground_truth, k)
    reciprocal_rank = 0.0
    for idx, item in enumerate(predicted[:max_k], start=1):
        if item in ground_truth:
            reciprocal_rank = 1.0 / idx
            break
    metrics["mrr"] = reciprocal_rank
    return metrics


def compute_ranking_metrics(
    per_user_predictions: dict[int, Sequence[int]],
    per_user_ground_truth: dict[int, set[int]],
    k_values: Iterable[int],
) -> RankingMetrics:
    recalls: dict[int, list[float]] = {k: [] for k in k_values}
    precisions: dict[int, list[float]] = {k: [] for k in k_values}
    ndcgs: dict[int, list[float]] = {k: [] for k in k_values}
    hit_rates: dict[int, list[float]] = {k: [] for k in k_values}
    maps: dict[int, list[float]] = {k: [] for k in k_values}
    per_user_results: list[dict[str, float]] = []
    mrr_scores: list[float] = []

    for user_idx, prediction in per_user_predictions.items():
        ground_truth = per_user_ground_truth.get(user_idx, set())
        if not ground_truth:
            continue
        metrics = per_user_metrics(prediction, ground_truth, k_values)
        per_user_results.append(metrics)
        for k in k_values:
            recalls[k].append(metrics[f"recall@{k}"])
            precisions[k].append(metrics[f"precision@{k}"])
            ndcgs[k].append(metrics[f"ndcg@{k}"])
            hit_rates[k].append(metrics[f"hit_rate@{k}"])
            maps[k].append(metrics[f"map@{k}"])
        mrr_scores.append(metrics["mrr"])

    def _aggregate(values: Iterable[float]) -> float:
        arr = list(values)
        if not arr:
            return 0.0
        return float(np.mean(arr))

    summary = RankingMetrics(
        recall={k: _aggregate(recalls[k]) for k in k_values},
        precision={k: _aggregate(precisions[k]) for k in k_values},
        ndcg={k: _aggregate(ndcgs[k]) for k in k_values},
        hit_rate={k: _aggregate(hit_rates[k]) for k in k_values},
        map={k: _aggregate(maps[k]) for k in k_values},
        mrr=_aggregate(mrr_scores),
        per_user=per_user_results,
    )
    return summary
