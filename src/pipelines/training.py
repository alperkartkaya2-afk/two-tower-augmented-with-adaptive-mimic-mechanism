"""
Training orchestration entry point.

This module couples data preparation, model construction, optimisation, model
evaluation, and lightweight recommendation previews. The implementation keeps
scripts thin while remaining testable and extensible.
"""

from __future__ import annotations

import json
import random
import time
from collections import Counter
from dataclasses import dataclass, field
from itertools import product
from pathlib import Path
from typing import Any, Iterable, Mapping, Sequence

import numpy as np
import pandas as pd
import torch
from loguru import logger
import torch.nn.functional as F
from torch import nn
from torch.utils.data import DataLoader
from torch.nn.utils import clip_grad_norm_

try:  # pragma: no cover - optional dependency
    import faiss
except ImportError:  # pragma: no cover - optional dependency
    faiss = None

from src.data import (
    InteractionDataset,
    build_training_dataset,
    load_dataset,
    sample_negative_items,
)
from src.data.features import parse_category_tokens
from src.models import AdaptiveMimicMechanism, TwoTowerModel, build_tower_encoder
from src.evaluation import (
    analyze_item_neighbors,
    compute_feature_correlations,
    compute_ranking_metrics,
    summarize_embedding_norms,
    summarize_user_alignment,
)
from src.reporting import save_loss_curves
from src.utils import clone_config, get_by_dotted_path, set_by_dotted_path


class DotProductSimilarity(nn.Module):
    """Similarity module that returns the dot product between embeddings."""

    def forward(
        self, user_embedding: torch.Tensor, item_embedding: torch.Tensor
    ) -> torch.Tensor:
        return (user_embedding * item_embedding).sum(dim=-1)


@dataclass
class TrainingHistory:
    train_loss: list[float] = field(default_factory=list)
    val_loss: list[float] = field(default_factory=list)
    test_loss: list[float] = field(default_factory=list)
    monitored_metric: list[float] = field(default_factory=list)


@dataclass
class TrainingResult:
    config: Mapping[str, Any]
    history: TrainingHistory
    runtime_seconds: float
    best_metric: float | None
    best_epoch: int | None
    best_checkpoint_path: Path | None
    val_metrics: Any | None
    test_metrics: Any | None
    overrides: Mapping[str, Any] | None = None
    loss_plot_path: Path | None = None
    embedding_summary_path: Path | None = None


@dataclass
class EarlyStoppingController:
    metric: str
    mode: str = "max"
    patience: int = 3
    min_delta: float = 0.0
    best_value: float | None = None
    best_epoch: int | None = None
    epochs_without_improvement: int = 0

    def update(self, value: float | None, epoch: int) -> bool:
        """Update the controller and return True if training should stop."""
        if value is None:
            # Nothing to monitor; disable early stopping behaviour.
            return False

        improved = False
        if self.best_value is None:
            improved = True
        elif self.mode == "max":
            improved = value > (self.best_value + self.min_delta)
        else:
            improved = value < (self.best_value - self.min_delta)

        if improved:
            self.best_value = value
            self.best_epoch = epoch
            self.epochs_without_improvement = 0
            return False

        self.epochs_without_improvement += 1
        return self.epochs_without_improvement >= max(self.patience, 1)


def _extract_metric_value(metrics_summary: Any, metric: str) -> float | None:
    if metrics_summary is None:
        return None

    metric = metric.lower()
    if "@" in metric:
        prefix, k_str = metric.split("@", 1)
        try:
            k = int(k_str)
        except ValueError:
            return None
        table = getattr(metrics_summary, prefix, None)
        if table is None:
            return None
        return table.get(k)

    value = getattr(metrics_summary, metric, None)
    if isinstance(value, (int, float)):
        return float(value)
    return None


def _clone_state_dict(module: nn.Module) -> dict[str, torch.Tensor]:
    return {key: param.detach().cpu().clone() for key, param in module.state_dict().items()}


def _load_state_dict(module: nn.Module, state: Mapping[str, torch.Tensor], device: torch.device) -> None:
    remapped = {key: tensor.to(device) if torch.is_tensor(tensor) else tensor for key, tensor in state.items()}
    module.load_state_dict(remapped)


def _save_checkpoint(
    directory: Path,
    *,
    experiment_name: str,
    epoch: int,
    model: TwoTowerModel,
    optimizers: Sequence[torch.optim.Optimizer],
    metric_name: str | None,
    metric_value: float | None,
    template: str | None = None,
) -> Path:
    directory.mkdir(parents=True, exist_ok=True)
    safe_metric = (metric_name or "metric").replace("@", "at").replace("/", "_")
    filename_template = template or "{experiment}_{metric}_epoch{epoch}.pt"
    value = metric_value if metric_value is not None else 0.0
    filename = filename_template.format(
        experiment=experiment_name,
        metric=safe_metric,
        value=value,
        epoch=epoch,
    )
    path = directory / filename

    state = {
        "epoch": epoch,
        "model_state_dict": model.state_dict(),
        "optimizer_state_dicts": [opt.state_dict() for opt in optimizers],
        "metric_name": metric_name,
        "metric_value": metric_value,
        "timestamp": time.time(),
    }
    torch.save(state, path)
    return path


def _seed_everything(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def _split_train_validation(
    interactions: pd.DataFrame,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """
    Split interactions into train/validation by holding out the latest record per user.
    """
    df = interactions.copy()
    if "timestamp" not in df.columns:
        logger.warning(
            "No timestamp column detected; skipping hold-out split and using all "
            "interactions for training."
        )
        return df, df.iloc[0:0]

    df = df.sort_values("timestamp").reset_index(drop=True)
    val_indices: list[int] = []

    for user_idx, group in df.groupby("user_idx"):
        valid_timestamps = group["timestamp"].dropna()
        if valid_timestamps.empty or len(group) <= 1:
            continue
        idx = valid_timestamps.idxmax()
        val_indices.append(int(idx))

    if not val_indices:
        logger.warning(
            "Validation split empty after hold-out; training will proceed without "
            "evaluation."
        )
        return df, df.iloc[0:0]

    val_df = df.loc[val_indices].reset_index(drop=True)
    train_df = df.drop(index=val_indices).reset_index(drop=True)
    return train_df, val_df


def _split_train_validation_test(
    interactions: pd.DataFrame,
    *,
    train_fraction: float | None,
    test_fraction: float | None,
    seed: int | None = None,
) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    train_df, val_df = _split_train_validation(interactions)

    if train_fraction is not None and test_fraction is None:
        test_fraction = max(0.0, 1.0 - float(train_fraction))

    test_fraction = float(test_fraction or 0.0)
    if test_fraction <= 0.0 or train_df.empty:
        return train_df, val_df, train_df.iloc[0:0]

    rng = np.random.default_rng(seed)
    test_size = max(1, int(round(len(train_df) * min(test_fraction, 1.0))))
    if test_size >= len(train_df):
        test_df = train_df.copy()
        train_df = train_df.iloc[0:0]
        return train_df.reset_index(drop=True), val_df, test_df.reset_index(drop=True)

    indices = train_df.index.to_numpy()
    sampled = rng.choice(indices, size=test_size, replace=False)
    test_df = train_df.loc[sampled].copy().reset_index(drop=True)
    train_df = train_df.drop(index=sampled).reset_index(drop=True)

    return train_df, val_df, test_df


def _build_dataloader(
    interactions: pd.DataFrame, batch_size: int, *, shuffle: bool = True
) -> DataLoader:
    dataset = InteractionDataset(interactions)
    return DataLoader(dataset, batch_size=batch_size, shuffle=shuffle, drop_last=False)


def _select_similarity(name: str) -> nn.Module:
    name = name.lower()
    if name == "cosine":
        return nn.CosineSimilarity(dim=-1)
    if name == "dot":
        return DotProductSimilarity()
    raise ValueError(f"Unsupported similarity function: {name}")


def _collect_parameter_groups(
    model: TwoTowerModel,
) -> tuple[list[torch.nn.Parameter], list[torch.nn.Parameter]]:
    dense_params: list[torch.nn.Parameter] = []
    sparse_params: list[torch.nn.Parameter] = []
    seen: set[int] = set()

    def add_param(param: torch.nn.Parameter, collection: list[torch.nn.Parameter]) -> None:
        pid = id(param)
        if pid in seen:
            return
        collection.append(param)
        seen.add(pid)

    def handle_encoder(encoder: nn.Module) -> None:
        embedding = getattr(encoder, "embedding", None)
        if isinstance(embedding, nn.Embedding):
            weight = embedding.weight
            if getattr(embedding, "sparse", False):
                add_param(weight, sparse_params)
            else:
                add_param(weight, dense_params)
        for name, param in encoder.named_parameters():
            if name == "embedding.weight":
                continue
            add_param(param, dense_params)

    handle_encoder(model.user_encoder)
    handle_encoder(model.item_encoder)

    for param in model.parameters():
        add_param(param, dense_params)

    return dense_params, sparse_params


def _build_user_profiles(training_dataset) -> dict[int, dict[str, set[str]]]:
    items_lookup = training_dataset.items.set_index("item_idx")
    profiles: dict[int, dict[str, set[str]]] = {}
    for user_idx, group in training_dataset.interactions.groupby("user_idx"):
        categories: set[str] = set()
        authors: set[str] = set()
        for item_idx in group["item_idx"]:
            if item_idx not in items_lookup.index:
                continue
            row = items_lookup.loc[item_idx]
            categories.update(parse_category_tokens(row.get("categories")))
            author = row.get("author")
            if isinstance(author, str) and author:
                authors.add(author.strip())
        profiles[int(user_idx)] = {"categories": categories, "authors": authors}
    return profiles


def _score_all_items_for_user(
    model: TwoTowerModel,
    *,
    user_idx: int,
    top_k: int,
    num_items: int,
    user_features: torch.Tensor | None,
    item_features: torch.Tensor | None,
    device: torch.device,
    batch_size: int = 50000,
) -> list[int]:
    user_tensor = torch.tensor([user_idx], dtype=torch.long, device=device)
    user_inputs = {"indices": user_tensor}
    if user_features is not None and user_features.numel() > 0:
        user_inputs["features"] = user_features.index_select(0, user_tensor)
    user_embedding = model.user_encoder(user_inputs)
    mimic_module = getattr(model, "adaptive_mimic", None)
    if mimic_module is not None:
        user_embedding = mimic_module.augment_users(user_tensor, user_embedding)

    top_scores = torch.empty(0, device=device)
    top_items = torch.empty(0, dtype=torch.long, device=device)

    with torch.no_grad():
        for start in range(0, num_items, batch_size):
            end = min(start + batch_size, num_items)
            batch_indices = torch.arange(start, end, dtype=torch.long, device=device)
            item_inputs = {"indices": batch_indices}
            if item_features is not None and item_features.numel() > 0:
                item_inputs["features"] = item_features.index_select(0, batch_indices)
            item_embeddings = model.item_encoder(item_inputs)
            if mimic_module is not None:
                item_embeddings = mimic_module.augment_items(batch_indices, item_embeddings)

            if isinstance(model.similarity, nn.CosineSimilarity):
                scores = model.similarity(
                    F.normalize(user_embedding, dim=-1),
                    F.normalize(item_embeddings, dim=-1),
                )
            else:
                scores = model.similarity(user_embedding, item_embeddings)

            batch_top = torch.topk(scores.squeeze(0), k=min(top_k, scores.shape[-1]))
            combined_scores = torch.cat([top_scores, batch_top.values])
            combined_items = torch.cat([top_items, batch_indices[batch_top.indices]])

            if combined_scores.shape[0] > top_k:
                global_top = torch.topk(combined_scores, k=top_k)
                top_scores = global_top.values
                top_items = combined_items[global_top.indices]
            else:
                top_scores = combined_scores
                top_items = combined_items

    return top_items.cpu().tolist()


def _write_recommendation_report(
    report_path: Path,
    *,
    metrics_summary,
    embedding_stats: dict[str, Any],
    recommendations: list[dict[str, Any]],
    loss_plot_path: Path | None = None,
    history: TrainingHistory | None = None,
    monitor_metric: str | None = None,
    best_epoch: int | None = None,
    feature_correlations: list[dict[str, float]] | None = None,
) -> None:
    report_path.parent.mkdir(parents=True, exist_ok=True)
    lines: list[str] = []
    lines.append("# Recommendation Evaluation Report\n")

    lines.append("## Ranking Metrics\n")
    for metric_name, values in [
        ("Recall", metrics_summary.recall),
        ("Precision", metrics_summary.precision),
        ("NDCG", metrics_summary.ndcg),
        ("Hit Rate", metrics_summary.hit_rate),
        ("MAP", metrics_summary.map),
    ]:
        lines.append(f"- **{metric_name}**: " + ", ".join(f"@{k}={v:.4f}" for k, v in values.items()))
    lines.append("")

    if loss_plot_path is not None:
        lines.append("## Loss Curves\n")
        relative_path = loss_plot_path.as_posix()
        lines.append(
            "Training, validation, and test losses tracked across epochs. Monitoring metric:"
        )
        if monitor_metric and best_epoch is not None:
            lines.append(f"- Best {monitor_metric} achieved at epoch {best_epoch}")
        lines.append(f"![Loss curves]({relative_path})\n")

        if history is not None:
            lines.append("Epoch | Train | Validation | Test")
            lines.append("--- | --- | --- | ---")
            epochs = range(1, len(history.train_loss) + 1)
            for idx, epoch in enumerate(epochs):
                train_loss = history.train_loss[idx]
                val_loss = history.val_loss[idx] if idx < len(history.val_loss) else float("nan")
                test_loss = history.test_loss[idx] if idx < len(history.test_loss) else float("nan")
                lines.append(f"{epoch} | {train_loss:.4f} | {val_loss:.4f} | {test_loss:.4f}")
            lines.append("")

    lines.append("## Embedding Diagnostics\n")
    user_norms = embedding_stats["user_norms"]
    item_norms = embedding_stats["item_norms"]
    lines.append(
        f"- User embedding norms: mean={user_norms['mean']:.4f}, std={user_norms['std']:.4f}, "
        f"min={user_norms['min']:.4f}, max={user_norms['max']:.4f}"
    )
    lines.append(
        f"- Item embedding norms: mean={item_norms['mean']:.4f}, std={item_norms['std']:.4f}, "
        f"min={item_norms['min']:.4f}, max={item_norms['max']:.4f}"
    )
    neighbor_stats = embedding_stats["item_neighbor_overlap"]
    lines.append(
        f"- Item neighbor category overlap (k={neighbor_stats.get('k', 'NA')}): "
        f"mean={neighbor_stats['category_overlap_mean']:.4f}, std={neighbor_stats['category_overlap_std']:.4f}"
    )
    alignment = embedding_stats["user_alignment"]
    lines.append(
        f"- User embedding vs. feature alignment (cosine): mean={alignment['cosine_mean']:.4f}, "
        f"std={alignment['cosine_std']:.4f}"
    )
    lines.append("")

    if feature_correlations:
        lines.append("### Feature Correlations\n")
        lines.append("Feature | Pearson r | p-value")
        lines.append("--- | --- | ---")
        for entry in feature_correlations:
            lines.append(
                f"{entry['feature']} | {entry['pearson_r']:.4f} | {entry['p_value']:.2e}"
            )
        lines.append("")

    lines.append("## Sample User Recommendations\n")
    for entry in recommendations:
        lines.append(f"- **User** `{entry['user_id']}` | category match {entry['category_match']:.2%} | "
                     f"author match {entry['author_match']:.2%}")
        lines.append(f"  - Historical categories: {', '.join(sorted(entry['history_categories'])[:5]) or 'N/A'}")
        for rank, rec in enumerate(entry["recommendations"], start=1):
            lines.append(
                f"  {rank}. {rec['title']} ({rec['asin']}) — "
                f"author: {rec['author'] or 'Unknown'} | categories: {', '.join(rec['categories']) or 'N/A'}"
            )
        lines.append("")

    report_path.write_text("\n".join(lines), encoding="utf-8")


def _write_embedding_summary(
    summary_path: Path,
    *,
    embedding_stats: dict[str, Any],
    mimic_stats: dict[str, Any],
    feature_correlations: list[dict[str, float]],
    monitor_metric: str | None,
    best_epoch: int | None,
) -> None:
    summary_path.parent.mkdir(parents=True, exist_ok=True)
    payload = {
        "embedding_stats": embedding_stats,
        "adaptive_mimic": mimic_stats,
        "feature_correlations": feature_correlations,
        "monitor_metric": monitor_metric,
        "best_epoch": best_epoch,
    }
    summary_path.write_text(json.dumps(payload, indent=2), encoding="utf-8")


def _write_benchmark_report(report_path: Path, results: Sequence[TrainingResult]) -> None:
    if not results:
        return

    report_path.parent.mkdir(parents=True, exist_ok=True)
    lines: list[str] = []
    lines.append("# Training Benchmark Summary\n")
    lines.append("Run | Overrides | Best Metric | Best Epoch | Runtime (s) | Optimizer | Embedding Dim")
    lines.append("--- | --- | --- | --- | --- | --- | ---")

    for idx, result in enumerate(results, start=1):
        overrides = ", ".join(f"{k}={v}" for k, v in (result.overrides or {}).items()) or "-"
        metric = result.best_metric if result.best_metric is not None else float("nan")
        optimizer = get_by_dotted_path(result.config, "training.optimizer", "adam")
        embed_dim = get_by_dotted_path(
            result.config,
            "model.user_encoder.id_embedding.params.embedding_dim",
            "?",
        )
        lines.append(
            f"{idx} | {overrides} | {metric:.4f} | {result.best_epoch or '-'} | "
            f"{result.runtime_seconds:.1f} | {optimizer} | {embed_dim}"
        )

    report_path.write_text("\n".join(lines), encoding="utf-8")


def _compute_covariance(matrix: torch.Tensor) -> torch.Tensor:
    if matrix.shape[0] <= 1:
        return torch.zeros(
            (matrix.shape[1], matrix.shape[1]),
            device=matrix.device,
            dtype=matrix.dtype,
        )
    centered = matrix - matrix.mean(dim=0, keepdim=True)
    return centered.T @ centered / (matrix.shape[0] - 1)


def _category_alignment_loss(
    item_indices: torch.Tensor,
    item_embeddings: torch.Tensor,
    *,
    category_tensor: torch.Tensor | None,
    major_category_id: int | None,
) -> torch.Tensor:
    if (
        category_tensor is None
        or major_category_id is None
        or item_indices.numel() == 0
    ):
        return item_embeddings.new_zeros(())

    batch_categories = category_tensor.index_select(0, item_indices)
    unique_categories = batch_categories.unique()
    if unique_categories.numel() <= 1:
        return item_embeddings.new_zeros(())

    major_mask = batch_categories == major_category_id
    if major_mask.sum() < 2:
        return item_embeddings.new_zeros(())

    major_cov = _compute_covariance(item_embeddings[major_mask])
    loss = item_embeddings.new_zeros(())
    compared = 0
    for cat_id in unique_categories.tolist():
        if cat_id == major_category_id:
            continue
        mask = batch_categories == cat_id
        if mask.sum() < 2:
            continue
        cov = _compute_covariance(item_embeddings[mask])
        diff = cov - major_cov
        loss = loss + torch.sum(diff * diff)
        compared += 1
    if compared == 0:
        return item_embeddings.new_zeros(())
    return loss / compared


def _build_item_category_tensor(
    items: pd.DataFrame,
    *,
    num_items: int,
    device: torch.device,
) -> tuple[torch.Tensor | None, int | None]:
    if num_items == 0 or "item_idx" not in items:
        return None, None

    categories = ["<unknown>"] * num_items
    counts: Counter[str] = Counter()
    for _, row in items.iterrows():
        idx = int(row["item_idx"])
        tokens = parse_category_tokens(row.get("categories"))
        primary = tokens[0] if tokens else "<unknown>"
        categories[idx] = primary
        counts[primary] += 1

    if not counts:
        return None, None

    category_to_id = {category: idx for idx, category in enumerate(counts.keys())}
    tensor = torch.tensor(
        [category_to_id[cat] for cat in categories],
        dtype=torch.long,
        device=device,
    )
    major_category = max(counts.items(), key=lambda kv: kv[1])[0]
    return tensor, category_to_id[major_category]


def _encode_item_embeddings(
    model: TwoTowerModel,
    *,
    num_items: int,
    item_features: torch.Tensor | None,
    device: torch.device,
    batch_size: int = 8192,
) -> torch.Tensor:
    if num_items == 0:
        return torch.empty((0, 0), dtype=torch.float32)

    model.eval()
    outputs: list[torch.Tensor] = []
    dim: int | None = None

    with torch.no_grad():
        for start in range(0, num_items, batch_size):
            end = min(start + batch_size, num_items)
            indices = torch.arange(start, end, device=device, dtype=torch.long)
            inputs = {"indices": indices}
            if item_features is not None and item_features.numel() > 0:
                inputs["features"] = item_features.index_select(0, indices)
            embedding = model.item_encoder(inputs)
            if getattr(model, "adaptive_mimic", None) is not None:
                embedding = model.adaptive_mimic.augment_items(indices, embedding)
            outputs.append(embedding.detach().cpu())
            dim = embedding.shape[-1]

    if not outputs:
        return torch.empty((0, dim or 0), dtype=torch.float32)
    return torch.cat(outputs, dim=0)


def _prepare_faiss_resources(
    model: TwoTowerModel,
    *,
    num_items: int,
    item_features: torch.Tensor | None,
    device: torch.device,
    similarity_module: nn.Module,
    batch_size: int,
    retain_embeddings: bool = False,
) -> dict[str, Any] | None:
    if faiss is None or num_items == 0:
        return None

    embeddings = _encode_item_embeddings(
        model,
        num_items=num_items,
        item_features=item_features,
        device=device,
        batch_size=batch_size,
    )
    if embeddings.numel() == 0:
        return None

    np_embeddings = embeddings.numpy().astype("float32", copy=False)
    normalize = isinstance(similarity_module, nn.CosineSimilarity)
    if normalize:
        faiss.normalize_L2(np_embeddings)

    index = faiss.IndexFlatIP(np_embeddings.shape[1])
    index.add(np_embeddings)
    payload: dict[str, Any] = {"index": index, "normalize": normalize}
    if retain_embeddings:
        payload["embeddings"] = np_embeddings
    return payload


def _save_faiss_artifacts(
    resources: dict[str, Any] | None,
    *,
    index_path: Path,
    embedding_path: Path,
) -> None:
    if faiss is None or resources is None:
        return
    index = resources.get("index")
    embeddings = resources.get("embeddings")
    if index is None or embeddings is None:
        return
    index_path.parent.mkdir(parents=True, exist_ok=True)
    embedding_path.parent.mkdir(parents=True, exist_ok=True)
    faiss.write_index(index, str(index_path))
    np.save(embedding_path, embeddings)


def _train_one_epoch(
    model: TwoTowerModel,
    dataloader: DataLoader,
    *,
    optimizers: list[torch.optim.Optimizer],
    criterion: nn.BCEWithLogitsLoss,
    negatives_per_positive: int,
    num_items: int,
    user_positive_items: dict[int, set[int]],
    user_features: torch.Tensor | None,
    item_features: torch.Tensor | None,
    device: torch.device,
    gradient_clip_norm: float | None = None,
    loss_weights: Mapping[str, Any] | None = None,
    item_category_tensor: torch.Tensor | None = None,
    major_category_id: int | None = None,
) -> float:
    model.train()
    running_loss = 0.0
    total_samples = 0
    mimic_module = getattr(model, "adaptive_mimic", None)
    loss_weights = loss_weights or {}
    lambda_mimic_user = float(loss_weights.get("mimic_user", 0.0))
    lambda_mimic_item = float(loss_weights.get("mimic_item", 0.0))
    lambda_cal = float(loss_weights.get("category_alignment", 0.0))

    for users_batch, pos_items_batch in dataloader:
        users_batch = users_batch.to(device)
        pos_items_batch = pos_items_batch.to(device)

        neg_items_batch = sample_negative_items(
            users_batch,
            num_items=num_items,
            positives=user_positive_items,
            num_negatives=negatives_per_positive,
            device=device,
        )

        for opt in optimizers:
            opt.zero_grad()

        user_inputs = {"indices": users_batch}
        if user_features is not None and user_features.numel() > 0:
            user_inputs["features"] = user_features.index_select(0, users_batch)

        pos_item_inputs = {"indices": pos_items_batch}
        if item_features is not None and item_features.numel() > 0:
            pos_item_inputs["features"] = item_features.index_select(0, pos_items_batch)

        user_embeddings_base = model.user_encoder(user_inputs)
        pos_item_embeddings_base = model.item_encoder(pos_item_inputs)

        if mimic_module is not None:
            (
                user_embeddings,
                pos_item_embeddings,
                mimic_user_loss,
                mimic_item_loss,
            ) = mimic_module(
                user_indices=users_batch,
                item_indices=pos_items_batch,
                user_embedding=user_embeddings_base,
                item_embedding=pos_item_embeddings_base,
            )
        else:
            user_embeddings = user_embeddings_base
            pos_item_embeddings = pos_item_embeddings_base
            mimic_user_loss = None
            mimic_item_loss = None

        pos_logits = (user_embeddings * pos_item_embeddings).sum(dim=-1)

        neg_flat = neg_items_batch.view(-1)
        neg_inputs = {"indices": neg_flat}
        if item_features is not None and item_features.numel() > 0:
            neg_inputs["features"] = item_features.index_select(0, neg_flat)
        neg_item_embeddings_base = model.item_encoder(neg_inputs)
        if mimic_module is not None:
            neg_item_embeddings = mimic_module.augment_items(
                neg_flat, neg_item_embeddings_base
            )
        else:
            neg_item_embeddings = neg_item_embeddings_base
        neg_item_embeddings = neg_item_embeddings.view(
            -1, negatives_per_positive, user_embeddings.shape[-1]
        )
        user_expanded = user_embeddings.unsqueeze(1)
        neg_logits = (user_expanded * neg_item_embeddings).sum(dim=-1)

        logits = torch.cat([pos_logits, neg_logits.reshape(-1)], dim=0)
        labels = torch.cat(
            [
                torch.ones_like(pos_logits, device=device),
                torch.zeros_like(neg_logits.reshape(-1), device=device),
            ],
            dim=0,
        )

        retrieval_loss = criterion(logits, labels)
        total_loss = retrieval_loss
        if mimic_user_loss is not None and lambda_mimic_user > 0:
            total_loss = total_loss + lambda_mimic_user * mimic_user_loss
        if mimic_item_loss is not None and lambda_mimic_item > 0:
            total_loss = total_loss + lambda_mimic_item * mimic_item_loss

        if lambda_cal > 0:
            combined_indices = torch.cat([pos_items_batch, neg_flat], dim=0)
            combined_embeddings = torch.cat(
                [
                    pos_item_embeddings,
                    neg_item_embeddings.reshape(-1, user_embeddings.shape[-1]),
                ],
                dim=0,
            )
            cal_loss = _category_alignment_loss(
                combined_indices,
                combined_embeddings,
                category_tensor=item_category_tensor,
                major_category_id=major_category_id,
            )
            total_loss = total_loss + lambda_cal * cal_loss

        total_loss.backward()

        if gradient_clip_norm is not None and gradient_clip_norm > 0:
            clip_grad_norm_(model.parameters(), gradient_clip_norm)
        for opt in optimizers:
            opt.step()

        batch_size = pos_logits.shape[0]
        running_loss += total_loss.item() * batch_size
        total_samples += batch_size

    return running_loss / max(total_samples, 1)


def _compute_loss(
    model: TwoTowerModel,
    dataloader: DataLoader,
    *,
    criterion: nn.BCEWithLogitsLoss,
    negatives_per_positive: int,
    num_items: int,
    user_positive_items: Mapping[int, set[int]],
    user_features: torch.Tensor | None,
    item_features: torch.Tensor | None,
    device: torch.device,
) -> float:
    model.eval()
    running_loss = 0.0
    total_samples = 0
    mimic_module = getattr(model, "adaptive_mimic", None)

    with torch.no_grad():
        for users_batch, pos_items_batch in dataloader:
            users_batch = users_batch.to(device)
            pos_items_batch = pos_items_batch.to(device)

            neg_items_batch = sample_negative_items(
                users_batch,
                num_items=num_items,
                positives=user_positive_items,
                num_negatives=negatives_per_positive,
                device=device,
            )

            user_inputs = {"indices": users_batch}
            if user_features is not None and user_features.numel() > 0:
                user_inputs["features"] = user_features.index_select(0, users_batch)

            pos_item_inputs = {"indices": pos_items_batch}
            if item_features is not None and item_features.numel() > 0:
                pos_item_inputs["features"] = item_features.index_select(0, pos_items_batch)

            user_embeddings = model.user_encoder(user_inputs)
            pos_item_embeddings = model.item_encoder(pos_item_inputs)
            if mimic_module is not None:
                user_embeddings = mimic_module.augment_users(users_batch, user_embeddings)
                pos_item_embeddings = mimic_module.augment_items(
                    pos_items_batch, pos_item_embeddings
                )
            pos_logits = (user_embeddings * pos_item_embeddings).sum(dim=-1)

            neg_flat = neg_items_batch.view(-1)
            neg_inputs = {"indices": neg_flat}
            if item_features is not None and item_features.numel() > 0:
                neg_inputs["features"] = item_features.index_select(0, neg_flat)
            neg_item_embeddings = model.item_encoder(neg_inputs)
            if mimic_module is not None:
                neg_item_embeddings = mimic_module.augment_items(
                    neg_flat, neg_item_embeddings
                )
            neg_item_embeddings = neg_item_embeddings.view(
                -1, negatives_per_positive, user_embeddings.shape[-1]
            )
            user_expanded = user_embeddings.unsqueeze(1)
            neg_logits = (user_expanded * neg_item_embeddings).sum(dim=-1)

            logits = torch.cat([pos_logits, neg_logits.reshape(-1)], dim=0)
            labels = torch.cat(
                [
                    torch.ones_like(pos_logits, device=device),
                    torch.zeros_like(neg_logits.reshape(-1), device=device),
                ],
                dim=0,
            )

            loss = criterion(logits, labels)
            batch_size = pos_logits.shape[0]
            running_loss += loss.item() * batch_size
            total_samples += batch_size

    if total_samples == 0:
        return 0.0
    return running_loss / total_samples


def _evaluate_model(
    model: TwoTowerModel,
    *,
    train_positive_map: dict[int, set[int]],
    val_interactions: pd.DataFrame,
    item_feature_tensor: torch.Tensor | None,
    user_feature_tensor: torch.Tensor | None,
    device: torch.device,
    num_items: int,
    candidate_samples: int,
    k_values: Iterable[int],
    rng: np.random.Generator,
    faiss_resources: dict[str, Any] | None = None,
    faiss_search_k: int = 0,
) -> tuple[dict[int, list[int]], dict[int, set[int]]]:
    if val_interactions.empty:
        return {}, {}

    model.eval()
    max_k = max(k_values)
    per_user_predictions: dict[int, list[int]] = {}
    per_user_ground_truth: dict[int, set[int]] = {}
    mimic_module = getattr(model, "adaptive_mimic", None)
    use_faiss = faiss_resources is not None and faiss is not None
    faiss_index = faiss_resources.get("index") if use_faiss else None
    faiss_normalize = bool(faiss_resources.get("normalize")) if use_faiss else False

    def _retrieve_with_faiss(
        user_embedding: torch.Tensor,
        blocked_items: set[int],
        ground_truth: set[int],
    ) -> list[int]:
        if faiss_index is None:
            return []
        query = user_embedding.detach().cpu().numpy().astype("float32", copy=False)
        if query.ndim == 1:
            query = query[None, :]
        if faiss_normalize:
            faiss.normalize_L2(query)
        search_limit = max(max_k + len(ground_truth), 1)
        search_k = max(faiss_search_k, search_limit + len(blocked_items))
        distances, indices = faiss_index.search(query, search_k)
        candidate_ids = indices[0].tolist()
        filtered: list[int] = []
        seen: set[int] = set()
        for item_id in candidate_ids:
            if item_id in blocked_items or item_id in seen or item_id < 0:
                continue
            filtered.append(int(item_id))
            seen.add(int(item_id))
            if len(filtered) >= search_limit:
                break
        for item_id in ground_truth:
            if item_id not in seen:
                filtered.append(item_id)
        return filtered[:max_k]

    def _retrieve_with_sampling(
        user_embedding: torch.Tensor,
        blocked_items: set[int],
        ground_truth: set[int],
    ) -> list[int]:
        candidates = set(ground_truth)
        available = list(set(range(num_items)) - blocked_items)
        if available:
            neg_budget = max(0, min(candidate_samples, len(available)))
            if neg_budget > 0:
                negatives = rng.choice(available, size=neg_budget, replace=False).tolist()
                candidates.update(int(n) for n in negatives)

        candidate_list = list(candidates)
        candidate_tensor = torch.tensor(candidate_list, device=device, dtype=torch.long)
        item_inputs = {"indices": candidate_tensor}
        if item_feature_tensor is not None and item_feature_tensor.numel() > 0:
            item_inputs["features"] = item_feature_tensor.index_select(0, candidate_tensor)

        candidate_embeddings = model.item_encoder(item_inputs)
        if mimic_module is not None:
            candidate_embeddings = mimic_module.augment_items(
                candidate_tensor, candidate_embeddings
            )

        if isinstance(model.similarity, nn.CosineSimilarity):
            scores = model.similarity(
                F.normalize(user_embedding, dim=-1),
                F.normalize(candidate_embeddings, dim=-1),
            )
        else:
            scores = model.similarity(user_embedding, candidate_embeddings)

        topk = torch.topk(scores.squeeze(0), k=min(max_k, len(candidate_list)))
        predicted_indices = topk.indices.cpu().tolist()
        return [candidate_list[i] for i in predicted_indices]

    with torch.no_grad():
        for user_idx, group in val_interactions.groupby("user_idx"):
            ground_truth = set(map(int, group["item_idx"].tolist()))
            if not ground_truth:
                continue

            per_user_ground_truth[int(user_idx)] = ground_truth

            user_tensor = torch.tensor([user_idx], device=device, dtype=torch.long)
            user_inputs = {"indices": user_tensor}
            if user_feature_tensor is not None and user_feature_tensor.numel() > 0:
                user_inputs["features"] = user_feature_tensor.index_select(0, user_tensor)

            user_embedding = model.user_encoder(user_inputs)
            if mimic_module is not None:
                user_embedding = mimic_module.augment_users(user_tensor, user_embedding)

            blocked_items = set(train_positive_map.get(int(user_idx), set()))
            if use_faiss:
                predictions = _retrieve_with_faiss(
                    user_embedding=user_embedding,
                    blocked_items=blocked_items,
                    ground_truth=ground_truth,
                )
            else:
                predictions = _retrieve_with_sampling(
                    user_embedding=user_embedding,
                    blocked_items=blocked_items,
                    ground_truth=ground_truth,
                )
            per_user_predictions[int(user_idx)] = predictions

    return per_user_predictions, per_user_ground_truth


def _log_recommendations(
    model: TwoTowerModel,
    training_dataset,
    *,
    sample_users: int,
    top_k: int,
    user_features: torch.Tensor | None,
    item_features: torch.Tensor | None,
    device: torch.device,
) -> list[dict[str, Any]]:
    results: list[dict[str, Any]] = []
    if sample_users <= 0:
        return results

    model.eval()
    num_users = len(training_dataset.user_mapping)
    num_items = len(training_dataset.item_mapping)
    if num_users == 0 or num_items == 0:
        return results

    user_indices = list(range(num_users))
    chosen_users = random.sample(user_indices, k=min(sample_users, len(user_indices)))

    items_df = training_dataset.items.set_index("item_idx")
    users_df = training_dataset.users.set_index("user_idx")
    user_profiles = _build_user_profiles(training_dataset)

    for user_idx in chosen_users:
        positives = training_dataset.user_positive_items.get(int(user_idx), set())
        recommended_items = _score_all_items_for_user(
            model,
            user_idx=user_idx,
            top_k=top_k + len(positives),
            num_items=num_items,
            user_features=user_features,
            item_features=item_features,
            device=device,
        )
        recommended_items = [
            item for item in recommended_items if item not in positives
        ][:top_k]

        user_record = users_df.loc[user_idx]
        display_user = user_record["userId"]
        profile = user_profiles.get(int(user_idx), {"categories": set(), "authors": set()})

        recommendations = []
        category_matches = 0
        author_matches = 0
        for item_idx in recommended_items:
            if item_idx not in items_df.index:
                continue
            item_row = items_df.loc[item_idx]
            categories = set(parse_category_tokens(item_row.get("categories")))
            author = item_row.get("author") if isinstance(item_row.get("author"), str) else ""
            if categories & profile["categories"]:
                category_matches += 1
            if author and author in profile["authors"]:
                author_matches += 1
            recommendations.append(
                {
                    "asin": item_row.get("parent_asin", ""),
                    "title": item_row.get("title", "<unknown>"),
                    "author": author,
                    "categories": sorted(categories)[:5],
                }
            )

        total = max(len(recommendations), 1)
        logger.info("User {} | Top {} recommendations:", display_user, len(recommendations))
        for rank, rec in enumerate(recommendations, start=1):
            logger.info(
                "  {}. {} ({}) | author={} | categories={}",
                rank,
                rec["title"],
                rec["asin"],
                rec["author"] or "Unknown",
                ", ".join(rec["categories"]) or "N/A",
            )

        results.append(
            {
                "user_id": display_user,
                "user_idx": int(user_idx),
                "recommendations": recommendations,
                "category_match": category_matches / total,
                "author_match": author_matches / total,
                "history_categories": profile["categories"],
                "history_authors": profile["authors"],
            }
        )
    return results


def _compute_mimic_statistics(
    mimic_module: AdaptiveMimicMechanism | None,
    *,
    user_indices: torch.Tensor,
    item_indices: torch.Tensor,
) -> dict[str, dict[str, float]]:
    stats: dict[str, dict[str, float]] = {"user": {}, "item": {}}
    if mimic_module is None:
        return stats

    with torch.no_grad():
        if user_indices.numel() > 0:
            user_aug = mimic_module.user_augmented(user_indices).detach().cpu()
            norms = user_aug.norm(dim=1)
            stats["user"] = {
                "mean_norm": float(norms.mean().item()),
                "std_norm": float(norms.std(unbiased=False).item()),
            }
        if item_indices.numel() > 0:
            item_aug = mimic_module.item_augmented(item_indices).detach().cpu()
            norms = item_aug.norm(dim=1)
            stats["item"] = {
                "mean_norm": float(norms.mean().item()),
                "std_norm": float(norms.std(unbiased=False).item()),
            }
    return stats


def _run_single_experiment(
    config: Mapping[str, Any],
    overrides: Mapping[str, Any] | None = None,
) -> TrainingResult:
    config = dict(config)
    experiment_cfg = dict(config.get("experiment", {}))
    seed = int(experiment_cfg.get("seed", 0))
    if "seed" in experiment_cfg:
        _seed_everything(seed)

    start_time = time.time()
    experiment_name = str(experiment_cfg.get("name", "experiment"))

    data_config = dict(config.get("data", {}))
    data_dir = Path(data_config.get("root", "data"))
    books_file = data_config.get("books_file")
    users_file = data_config.get("users_file")
    feature_params = data_config.get("feature_params", {})
    books_limit = data_config.get("books_limit")
    interactions_limit = data_config.get("interactions_limit")
    min_user_interactions = int(data_config.get("min_user_interactions", 0))
    min_item_interactions = int(data_config.get("min_item_interactions", 0))

    logger.info("Loading raw datasets from {}", data_dir)
    dataset = load_dataset(
        data_dir,
        books_file=books_file,
        interactions_file=users_file,
        books_limit=books_limit,
        interactions_limit=interactions_limit,
    )

    logger.info("Building training dataset for stage='train'")
    training_dataset = build_training_dataset(
        dataset,
        stage="train",
        feature_config=feature_params,
        min_user_interactions=min_user_interactions,
        min_item_interactions=min_item_interactions,
    )

    logger.debug(
        "Training dataset summary | users={} items={} interactions={} | feature_dim(item={}, user={})",
        len(training_dataset.user_mapping),
        len(training_dataset.item_mapping),
        len(training_dataset.interactions),
        training_dataset.item_feature_matrix.shape[1],
        training_dataset.user_feature_matrix.shape[1],
    )

    train_interactions, val_interactions, test_interactions = _split_train_validation_test(
        training_dataset.interactions,
        train_fraction=data_config.get("train_fraction"),
        test_fraction=data_config.get("test_fraction"),
        seed=seed,
    )
    logger.info(
        "Split interactions | train={} validation={} test={}",
        len(train_interactions),
        len(val_interactions),
        len(test_interactions),
    )

    training_config = dict(config.get("training", {}))
    batch_size = int(training_config.get("batch_size", 512))
    num_epochs = int(training_config.get("num_epochs", 10))
    learning_rate = float(training_config.get("learning_rate", 1e-3))
    weight_decay = float(training_config.get("weight_decay", 0.0))
    negatives_per_positive = int(training_config.get("negatives_per_positive", 5))
    gradient_clip_norm = training_config.get("gradient_clip_norm")
    if gradient_clip_norm is not None:
        gradient_clip_norm = float(gradient_clip_norm)
    loss_weights = dict(training_config.get("loss_weights", {}))

    model_config = dict(config.get("model", {}))
    device_name = model_config.get("device", "cpu")
    device = torch.device(device_name)
    item_category_tensor, major_category_id = _build_item_category_tensor(
        training_dataset.items,
        num_items=len(training_dataset.item_mapping),
        device=device,
    )

    user_encoder_cfg = model_config.get("user_encoder", {})
    item_encoder_cfg = model_config.get("item_encoder", {})

    user_feature_tensor = None
    if training_dataset.user_feature_matrix.size:
        user_feature_tensor = torch.from_numpy(
            training_dataset.user_feature_matrix
        ).to(device)

    item_feature_tensor = None
    if training_dataset.item_feature_matrix.size:
        item_feature_tensor = torch.from_numpy(
            training_dataset.item_feature_matrix
        ).to(device)

    user_encoder = build_tower_encoder(
        user_encoder_cfg,
        num_embeddings=len(training_dataset.user_mapping),
        feature_dim=training_dataset.user_feature_matrix.shape[1],
        device=device,
    )
    item_encoder = build_tower_encoder(
        item_encoder_cfg,
        num_embeddings=len(training_dataset.item_mapping),
        feature_dim=training_dataset.item_feature_matrix.shape[1],
        device=device,
    )

    similarity_module = _select_similarity(model_config.get("similarity", "cosine"))

    adaptive_mimic_cfg = dict(model_config.get("adaptive_mimic", {}))
    mimic_enabled = adaptive_mimic_cfg.get("enabled", True)
    adaptive_mimic_module: AdaptiveMimicMechanism | None = None
    if mimic_enabled:
        user_dim = getattr(user_encoder, "output_dim", user_encoder.embedding.embedding_dim)
        item_dim = getattr(item_encoder, "output_dim", item_encoder.embedding.embedding_dim)
        if user_dim != item_dim:
            raise ValueError("Adaptive mimic requires user and item embedding dimensions to match.")
        adaptive_mimic_module = AdaptiveMimicMechanism(
            num_users=len(training_dataset.user_mapping),
            num_items=len(training_dataset.item_mapping),
            embedding_dim=user_dim,
            init_std=float(adaptive_mimic_cfg.get("init_std", 0.02)),
        ).to(device)

    model = TwoTowerModel(
        user_encoder=user_encoder,
        item_encoder=item_encoder,
        similarity=similarity_module,
        adaptive_mimic=adaptive_mimic_module,
    ).to(device)

    logger.info(
        "Tower configuration | user_dim={} item_dim={} | user_features={} item_features={}",
        getattr(user_encoder, "output_dim", user_encoder.embedding.embedding_dim),
        getattr(item_encoder, "output_dim", item_encoder.embedding.embedding_dim),
        training_dataset.user_feature_matrix.shape[1],
        training_dataset.item_feature_matrix.shape[1],
    )

    optimizer_name = training_config.get("optimizer", "adam").lower()
    dense_params, sparse_params = _collect_parameter_groups(model)
    optimizers: list[torch.optim.Optimizer] = []

    if dense_params:
        if optimizer_name == "adam":
            dense_optimizer = torch.optim.Adam(
                dense_params, lr=learning_rate, weight_decay=weight_decay
            )
        elif optimizer_name == "adamw":
            dense_optimizer = torch.optim.AdamW(
                dense_params, lr=learning_rate, weight_decay=weight_decay
            )
        elif optimizer_name == "sgd":
            dense_optimizer = torch.optim.SGD(
                dense_params,
                lr=learning_rate,
                weight_decay=weight_decay,
                momentum=float(training_config.get("momentum", 0.0)),
            )
        else:
            raise ValueError(f"Unsupported optimizer: {optimizer_name}")
        optimizers.append(dense_optimizer)
    else:
        logger.warning(
            "No dense parameters detected for optimizer '{}'. Using sparse-only optimisation.",
            optimizer_name,
        )

    if sparse_params:
        sparse_optimizer = torch.optim.SparseAdam(
            sparse_params,
            lr=learning_rate,
            betas=tuple(training_config.get("betas", (0.9, 0.999))),
        )
        optimizers.append(sparse_optimizer)
        logger.info(
            "Using SparseAdam for {} sparse parameter tensors.",
            len(sparse_params),
        )

    if not optimizers:
        logger.error("No parameters were provided to optimizers. Aborting training.")
        return TrainingResult(
            config=config,
            history=TrainingHistory(),
            runtime_seconds=time.time() - start_time,
            best_metric=None,
            best_epoch=None,
            best_checkpoint_path=None,
            val_metrics=None,
            test_metrics=None,
            overrides=overrides,
        )

    criterion = nn.BCEWithLogitsLoss()

    if train_interactions.empty:
        logger.warning("No training interactions available; exiting early.")
        return TrainingResult(
            config=config,
            history=TrainingHistory(),
            runtime_seconds=time.time() - start_time,
            best_metric=None,
            best_epoch=None,
            best_checkpoint_path=None,
            val_metrics=None,
            test_metrics=None,
            overrides=overrides,
        )

    train_loader = _build_dataloader(train_interactions, batch_size=batch_size)
    val_loader = (
        _build_dataloader(val_interactions, batch_size=batch_size, shuffle=False)
        if not val_interactions.empty
        else None
    )
    test_loader = (
        _build_dataloader(test_interactions, batch_size=batch_size, shuffle=False)
        if not test_interactions.empty
        else None
    )

    eval_cfg = dict(config.get("evaluation", {}))
    metrics_k = eval_cfg.get("metrics_k", [10])
    if isinstance(metrics_k, int):
        metrics_k = [metrics_k]
    candidate_samples = int(eval_cfg.get("candidate_samples", 500))
    max_metric_k = max(metrics_k) if metrics_k else 10
    faiss_cfg = dict(eval_cfg.get("faiss", {}))
    faiss_enabled = bool(faiss_cfg.get("enabled", True))
    faiss_search_multiplier = int(faiss_cfg.get("search_k_multiplier", 4))
    faiss_batch_size = int(faiss_cfg.get("batch_size", 8192))
    faiss_index_path = Path(faiss_cfg.get("index_path", "artifacts/faiss/items.index"))
    faiss_embedding_path = Path(
        faiss_cfg.get("embedding_path", "artifacts/faiss/item_embeddings.npy")
    )
    if faiss_enabled and faiss is None:
        logger.warning(
            "FAISS is not available; falling back to brute-force retrieval metrics."
        )
        faiss_enabled = False
    faiss_search_k_value = max(
        max_metric_k * max(1, faiss_search_multiplier), max_metric_k
    )

    diag_cfg = dict(config.get("diagnostics", {}))
    item_sample_size = int(diag_cfg.get("item_sample_size", 500))
    user_sample_size = int(diag_cfg.get("user_sample_size", 5000))
    neighbor_k = int(diag_cfg.get("neighbor_k", 10))
    report_path = Path(diag_cfg.get("report_path", "artifacts/reports/recommendation_report.md"))
    loss_plot_target = Path(diag_cfg.get("loss_plot_path", "artifacts/reports/loss_curve.png"))
    embedding_summary_path = Path(
        diag_cfg.get("embedding_summary_path", "artifacts/reports/embedding_diagnostics.json")
    )
    feature_corr_top_k = int(diag_cfg.get("feature_corr_top_k", 15))

    monitor_cfg = training_config.get("early_stopping", {})
    monitor_metric = monitor_cfg.get("metric") if monitor_cfg.get("enabled", False) else None
    monitor_mode = str(monitor_cfg.get("mode", "max")).lower()
    patience = int(monitor_cfg.get("patience", 3))
    min_delta = float(monitor_cfg.get("min_delta", 0.0))
    early_controller = None
    if monitor_metric:
        if monitor_mode not in {"max", "min"}:
            raise ValueError("early_stopping.mode must be either 'max' or 'min'")
        early_controller = EarlyStoppingController(
            metric=str(monitor_metric),
            mode=monitor_mode,
            patience=patience,
            min_delta=min_delta,
        )

    checkpoint_cfg = training_config.get("checkpointing", {})
    checkpoint_enabled = bool(checkpoint_cfg.get("enabled", False))
    checkpoint_dir = Path(checkpoint_cfg.get("dir", "artifacts/checkpoints"))
    checkpoint_template = str(
        checkpoint_cfg.get(
            "filename_template", "{experiment}_{metric}_{value:.4f}_epoch{epoch}.pt"
        )
    )
    save_best_only = bool(checkpoint_cfg.get("save_best_only", True))
    keep_last = bool(checkpoint_cfg.get("keep_last", True))
    best_checkpoint_path: Path | None = None
    last_checkpoint_path: Path | None = None

    train_positive_map = {
        int(user_idx): set(map(int, group["item_idx"].tolist()))
        for user_idx, group in train_interactions.groupby("user_idx")
    }

    num_items = len(training_dataset.item_mapping)
    rng_seed = seed or 0

    history = TrainingHistory()
    best_metric_value: float | None = None
    best_epoch: int | None = None
    best_val_metrics = None
    best_test_metrics = None
    best_state_dict: dict[str, torch.Tensor] | None = None
    last_val_metrics = None
    last_test_metrics = None

    for epoch in range(1, num_epochs + 1):
        avg_loss = _train_one_epoch(
            model,
            train_loader,
            optimizers=optimizers,
            criterion=criterion,
            negatives_per_positive=negatives_per_positive,
            num_items=num_items,
            user_positive_items=training_dataset.user_positive_items,
            user_features=user_feature_tensor,
            item_features=item_feature_tensor,
            device=device,
            gradient_clip_norm=gradient_clip_norm,
            loss_weights=loss_weights,
            item_category_tensor=item_category_tensor,
            major_category_id=major_category_id,
        )
        history.train_loss.append(float(avg_loss))
        logger.info("Epoch {:03d}/{:03d} | train_loss={:.4f}", epoch, num_epochs, avg_loss)

        val_loss_value = float("nan")
        val_metrics = None
        monitor_value: float | None = None

        faiss_resources = None
        if faiss_enabled and (val_loader is not None or test_loader is not None):
            faiss_resources = _prepare_faiss_resources(
                model,
                num_items=num_items,
                item_features=item_feature_tensor,
                device=device,
                similarity_module=similarity_module,
                batch_size=faiss_batch_size,
            )

        if val_loader is not None:
            val_loss_value = _compute_loss(
                model,
                val_loader,
                criterion=criterion,
                negatives_per_positive=negatives_per_positive,
                num_items=num_items,
                user_positive_items=training_dataset.user_positive_items,
                user_features=user_feature_tensor,
                item_features=item_feature_tensor,
                device=device,
            )
            rng = np.random.default_rng(rng_seed * 997 + epoch)
            val_predictions, val_ground_truth = _evaluate_model(
                model,
                train_positive_map=train_positive_map,
                val_interactions=val_interactions,
                item_feature_tensor=item_feature_tensor,
                user_feature_tensor=user_feature_tensor,
                device=device,
                num_items=num_items,
                candidate_samples=candidate_samples,
                k_values=metrics_k,
                rng=rng,
                faiss_resources=faiss_resources,
                faiss_search_k=faiss_search_k_value,
            )
            val_metrics = compute_ranking_metrics(
                val_predictions, val_ground_truth, metrics_k
            )
            last_val_metrics = val_metrics
            for k in metrics_k:
                logger.info(
                    "Validation metrics @{} | recall={:.4f} precision={:.4f} ndcg={:.4f} hit_rate={:.4f} map={:.4f}",
                    k,
                    val_metrics.recall[k],
                    val_metrics.precision[k],
                    val_metrics.ndcg[k],
                    val_metrics.hit_rate[k],
                    val_metrics.map[k],
                )
            if monitor_metric:
                monitor_value = _extract_metric_value(val_metrics, str(monitor_metric))

        if test_loader is not None:
            test_loss_value = _compute_loss(
                model,
                test_loader,
                criterion=criterion,
                negatives_per_positive=negatives_per_positive,
                num_items=num_items,
                user_positive_items=training_dataset.user_positive_items,
                user_features=user_feature_tensor,
                item_features=item_feature_tensor,
                device=device,
            )
            history.test_loss.append(float(test_loss_value))
            rng = np.random.default_rng(rng_seed * 199 + epoch)
            test_predictions, test_ground_truth = _evaluate_model(
                model,
                train_positive_map=train_positive_map,
                val_interactions=test_interactions,
                item_feature_tensor=item_feature_tensor,
                user_feature_tensor=user_feature_tensor,
                device=device,
                num_items=num_items,
                candidate_samples=candidate_samples,
                k_values=metrics_k,
                rng=rng,
                faiss_resources=faiss_resources,
                faiss_search_k=faiss_search_k_value,
            )
            last_test_metrics = compute_ranking_metrics(
                test_predictions, test_ground_truth, metrics_k
            )
        else:
            history.test_loss.append(float("nan"))

        history.val_loss.append(float(val_loss_value))

        if monitor_metric and monitor_value is not None and early_controller is not None:
            should_stop = early_controller.update(monitor_value, epoch)
            improved = early_controller.best_epoch == epoch
            if improved:
                best_metric_value = early_controller.best_value
                best_epoch = epoch
        else:
            candidate_value = (
                val_loss_value if not np.isnan(val_loss_value) else avg_loss
            )
            should_stop = False
            improved = (
                best_metric_value is None
                or candidate_value < (best_metric_value - min_delta)
            )
            if improved:
                best_metric_value = float(candidate_value)
                best_epoch = epoch

        tracked_value = monitor_value
        if tracked_value is None:
            if best_metric_value is not None:
                tracked_value = best_metric_value
            elif not np.isnan(val_loss_value):
                tracked_value = val_loss_value
            else:
                tracked_value = avg_loss
        history.monitored_metric.append(float(tracked_value) if tracked_value is not None else float("nan"))

        if improved:
            best_state_dict = _clone_state_dict(model)
            best_val_metrics = val_metrics or last_val_metrics
            best_test_metrics = last_test_metrics
            if checkpoint_enabled:
                metric_for_checkpoint = (
                    monitor_value
                    if monitor_metric and monitor_value is not None
                    else (best_metric_value if best_metric_value is not None else avg_loss)
                )
                checkpoint_path = _save_checkpoint(
                    checkpoint_dir,
                    experiment_name=experiment_name,
                    epoch=epoch,
                    model=model,
                    optimizers=optimizers,
                    metric_name=str(monitor_metric) if monitor_metric else "loss",
                    metric_value=metric_for_checkpoint,
                    template=checkpoint_template,
                )
                best_checkpoint_path = checkpoint_path

        if checkpoint_enabled and not save_best_only:
            _save_checkpoint(
                checkpoint_dir,
                experiment_name=experiment_name,
                epoch=epoch,
                model=model,
                optimizers=optimizers,
                metric_name="epoch",
                metric_value=float(epoch),
                template=checkpoint_template,
            )

        if checkpoint_enabled and keep_last:
            last_checkpoint_path = _save_checkpoint(
                checkpoint_dir,
                experiment_name=experiment_name,
                epoch=epoch,
                model=model,
                optimizers=optimizers,
                metric_name="last",
                metric_value=float(epoch),
                template="{experiment}_last.pt",
            )

        if should_stop:
            logger.info(
                "Early stopping triggered after {} epochs without improvement.",
                patience,
            )
            break

    if best_state_dict is not None:
        _load_state_dict(model, best_state_dict, device)
    elif last_checkpoint_path is not None and best_checkpoint_path is None:
        best_checkpoint_path = last_checkpoint_path

    if best_val_metrics is None:
        best_val_metrics = last_val_metrics
    if best_val_metrics is None:
        best_val_metrics = compute_ranking_metrics({}, {}, metrics_k)
    if best_test_metrics is None:
        best_test_metrics = last_test_metrics
    if best_test_metrics is None:
        best_test_metrics = compute_ranking_metrics({}, {}, metrics_k)

    if best_metric_value is None and history.train_loss:
        best_metric_value = history.train_loss[-1]
        best_epoch = best_epoch or len(history.train_loss)

    items_df = training_dataset.items.set_index("item_idx")
    item_sample_indices = (
        torch.tensor(
            random.sample(range(num_items), k=min(item_sample_size, num_items)),
            dtype=torch.long,
            device=device,
        )
        if num_items > 0 and item_sample_size > 0
        else torch.empty(0, dtype=torch.long, device=device)
    )
    if item_sample_indices.numel() > 0:
        item_inputs = {"indices": item_sample_indices}
        if item_feature_tensor is not None and item_feature_tensor.numel() > 0:
            item_inputs["features"] = item_feature_tensor.index_select(0, item_sample_indices)
        item_sample_embeddings = model.item_encoder(item_inputs)
        if getattr(model, "adaptive_mimic", None) is not None:
            item_sample_embeddings = model.adaptive_mimic.augment_items(
                item_sample_indices, item_sample_embeddings
            )
        item_sample_embeddings = item_sample_embeddings.detach().cpu()
        item_sample_frame = items_df.loc[item_sample_indices.cpu().numpy()].reset_index(drop=True)
        item_feature_subset = training_dataset.item_feature_matrix[
            item_sample_indices.cpu().numpy()
        ]
    else:
        item_sample_embeddings = torch.empty((0, model.item_encoder.output_dim), dtype=torch.float32)
        item_sample_frame = items_df.iloc[0:0]
        item_feature_subset = np.zeros((0, training_dataset.item_feature_matrix.shape[1]))

    num_users = len(training_dataset.user_mapping)
    user_sample_indices = (
        torch.tensor(
            random.sample(range(num_users), k=min(user_sample_size, num_users)),
            dtype=torch.long,
            device=device,
        )
        if num_users > 0 and user_sample_size > 0
        else torch.empty(0, dtype=torch.long, device=device)
    )
    if user_sample_indices.numel() > 0:
        user_inputs = {"indices": user_sample_indices}
        if user_feature_tensor is not None and user_feature_tensor.numel() > 0:
            user_inputs["features"] = user_feature_tensor.index_select(0, user_sample_indices)
        user_sample_embeddings = model.user_encoder(user_inputs)
        if getattr(model, "adaptive_mimic", None) is not None:
            user_sample_embeddings = model.adaptive_mimic.augment_users(
                user_sample_indices, user_sample_embeddings
            )
        user_sample_embeddings = user_sample_embeddings.detach().cpu()
        user_feature_subset = (
            training_dataset.user_feature_matrix[user_sample_indices.cpu().numpy()]
            if training_dataset.user_feature_matrix.size
            else np.zeros((user_sample_indices.numel(), 0), dtype=np.float32)
        )
    else:
        user_sample_embeddings = torch.empty((0, model.user_encoder.output_dim), dtype=torch.float32)
        user_feature_subset = np.zeros((0, 0), dtype=np.float32)

    embedding_stats = {
        "user_norms": summarize_embedding_norms(user_sample_embeddings, label="user"),
        "item_norms": summarize_embedding_norms(item_sample_embeddings, label="item"),
        "item_neighbor_overlap": analyze_item_neighbors(
            item_sample_embeddings,
            item_sample_frame.reset_index(drop=True),
            k=neighbor_k,
            sample_size=item_sample_frame.shape[0],
        ),
        "user_alignment": summarize_user_alignment(
            user_sample_embeddings, user_feature_subset
        ),
    }

    mimic_stats = _compute_mimic_statistics(
        getattr(model, "adaptive_mimic", None),
        user_indices=user_sample_indices,
        item_indices=item_sample_indices,
    )

    feature_correlations: list[dict[str, float]] = []
    if item_feature_subset.size > 0:
        feature_names = training_dataset.feature_metadata.feature_names()
        scores = item_sample_embeddings.norm(dim=1).numpy()
        feature_correlations = compute_feature_correlations(
            item_feature_subset,
            scores,
            feature_names[: item_feature_subset.shape[1]],
            top_k=feature_corr_top_k,
        )

    rec_cfg = dict(config.get("recommendations", {}))
    sample_users = int(rec_cfg.get("sample_users", 3))
    rec_top_k = int(rec_cfg.get("top_k", 5))
    recommendation_samples = _log_recommendations(
        model,
        training_dataset,
        sample_users=sample_users,
        top_k=rec_top_k,
        user_features=user_feature_tensor,
        item_features=item_feature_tensor,
        device=device,
    )

    if faiss_enabled:
        final_faiss_resources = _prepare_faiss_resources(
            model,
            num_items=num_items,
            item_features=item_feature_tensor,
            device=device,
            similarity_module=similarity_module,
            batch_size=faiss_batch_size,
            retain_embeddings=True,
        )
        _save_faiss_artifacts(
            final_faiss_resources,
            index_path=faiss_index_path,
            embedding_path=faiss_embedding_path,
        )

    loss_plot_path: Path | None = None
    loss_series = {
        "Train": history.train_loss,
        "Validation": history.val_loss,
        "Test": history.test_loss,
    }
    if any(len(values) for values in loss_series.values()):
        try:
            loss_plot_path = save_loss_curves(loss_series, output_path=loss_plot_target)
        except ValueError:
            loss_plot_path = None

    _write_recommendation_report(
        report_path,
        metrics_summary=best_val_metrics,
        embedding_stats=embedding_stats,
        recommendations=recommendation_samples,
        loss_plot_path=loss_plot_path,
        history=history,
        monitor_metric=str(monitor_metric) if monitor_metric else "val_loss",
        best_epoch=best_epoch,
        feature_correlations=feature_correlations,
    )

    _write_embedding_summary(
        embedding_summary_path,
        embedding_stats=embedding_stats,
        mimic_stats=mimic_stats,
        feature_correlations=feature_correlations,
        monitor_metric=str(monitor_metric) if monitor_metric else "val_loss",
        best_epoch=best_epoch,
    )

    runtime = time.time() - start_time

    return TrainingResult(
        config=config,
        history=history,
        runtime_seconds=runtime,
        best_metric=best_metric_value,
        best_epoch=best_epoch,
        best_checkpoint_path=best_checkpoint_path,
        val_metrics=best_val_metrics,
        test_metrics=best_test_metrics,
        overrides=overrides,
        loss_plot_path=loss_plot_path,
        embedding_summary_path=embedding_summary_path,
    )


def _run_experiment_grid(
    config: Mapping[str, Any],
    grid: Mapping[str, Sequence[Any]],
) -> list[TrainingResult]:
    if not grid:
        return [_run_single_experiment(config)]

    keys = list(grid.keys())
    values_product = list(product(*[grid[key] for key in keys]))
    results: list[TrainingResult] = []

    for idx, combination in enumerate(values_product):
        overrides = dict(zip(keys, combination))
        run_config = clone_config(config)
        for key, value in overrides.items():
            set_by_dotted_path(run_config, key, value)
        run_config.setdefault("experiment", {})
        base_name = str(config.get("experiment", {}).get("name", "experiment"))
        run_config["experiment"]["name"] = f"{base_name}_sweep{idx:02d}"
        result = _run_single_experiment(run_config, overrides=overrides)
        results.append(result)

    return results


def run_training(config: Mapping[str, Any]) -> list[TrainingResult] | TrainingResult:
    experiment_cfg = config.get("experiment", {})
    grid = experiment_cfg.get("grid") or {}

    if grid:
        results = _run_experiment_grid(config, grid)
    else:
        results = [_run_single_experiment(config)]

    benchmark_path = experiment_cfg.get("benchmark_report")
    if benchmark_path:
        _write_benchmark_report(Path(benchmark_path), results)

    if len(results) == 1:
        return results[0]
    return results
