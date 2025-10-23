"""
Training orchestration entry point.

This module couples data preparation, model construction, optimisation, model
evaluation, and lightweight recommendation previews. The implementation keeps
scripts thin while remaining testable and extensible.
"""

from __future__ import annotations

import random
from pathlib import Path
from typing import Any, Iterable, Mapping

import numpy as np
import pandas as pd
import torch
from loguru import logger
from torch import nn
import torch.nn.functional as F
from torch.utils.data import DataLoader

from src.data import (
    InteractionDataset,
    build_training_dataset,
    load_dataset,
    sample_negative_items,
)
from src.data.features import parse_category_tokens
from src.models import TwoTowerModel, build_tower_encoder
from src.evaluation import (
    analyze_item_neighbors,
    compute_feature_correlations,
    compute_ranking_metrics,
    summarize_embedding_norms,
    summarize_user_alignment,
)


class DotProductSimilarity(nn.Module):
    """Similarity module that returns the dot product between embeddings."""

    def forward(
        self, user_embedding: torch.Tensor, item_embedding: torch.Tensor
    ) -> torch.Tensor:
        return (user_embedding * item_embedding).sum(dim=-1)


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
) -> float:
    model.train()
    running_loss = 0.0
    total_samples = 0

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

        user_embeddings = model.user_encoder(user_inputs)
        pos_item_embeddings = model.item_encoder(pos_item_inputs)
        pos_logits = (user_embeddings * pos_item_embeddings).sum(dim=-1)

        neg_flat = neg_items_batch.view(-1)
        neg_inputs = {"indices": neg_flat}
        if item_features is not None and item_features.numel() > 0:
            neg_inputs["features"] = item_features.index_select(0, neg_flat)
        neg_item_embeddings = model.item_encoder(neg_inputs).view(
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
        loss.backward()
        for opt in optimizers:
            opt.step()

        batch_size = pos_logits.shape[0]
        running_loss += loss.item() * batch_size
        total_samples += batch_size

    return running_loss / max(total_samples, 1)


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
) -> tuple[dict[int, list[int]], dict[int, set[int]]]:
    if val_interactions.empty:
        return {}, {}

    model.eval()
    max_k = max(k_values)
    per_user_predictions: dict[int, list[int]] = {}
    per_user_ground_truth: dict[int, set[int]] = {}

    with torch.no_grad():
        for user_idx, group in val_interactions.groupby("user_idx"):
            ground_truth = set(map(int, group["item_idx"].tolist()))
            if not ground_truth:
                continue

            per_user_ground_truth[int(user_idx)] = ground_truth

            #blocked_items = set(train_positive_map.get(int(user_idx), set()))
            #blocked_items.update(ground_truth)
            #candidates = set(ground_truth)

            #while len(candidates) - len(ground_truth) < candidate_samples:
            #    sampled = int(rng.integers(num_items))
            #    if sampled in blocked_items or sampled in candidates:
            #        continue
            #    candidates.add(sampled)

            #candidate_list = list(candidates)
            
            blocked_items = set(train_positive_map.get(int(user_idx), set()))
            blocked_items.update(ground_truth)

            remaining = list(set(range(num_items)) - blocked_items)
            neg_budget = max(0, min(candidate_samples, len(remaining)))
            if neg_budget > 0:
                negatives = rng.choice(remaining, size=neg_budget, replace=False).tolist()
            else:
                negatives = []

            candidates = set(ground_truth)
            candidates.update(negatives)
            candidate_list = list(candidates)
            
            candidate_tensor = torch.tensor(candidate_list, device=device, dtype=torch.long)
            item_inputs = {"indices": candidate_tensor}
            if item_feature_tensor is not None and item_feature_tensor.numel() > 0:
                item_inputs["features"] = item_feature_tensor.index_select(0, candidate_tensor)

            user_tensor = torch.tensor([user_idx], device=device, dtype=torch.long)
            user_inputs = {"indices": user_tensor}
            if user_feature_tensor is not None and user_feature_tensor.numel() > 0:
                user_inputs["features"] = user_feature_tensor.index_select(0, user_tensor)

            user_embedding = model.user_encoder(user_inputs)
            candidate_embeddings = model.item_encoder(item_inputs)

            if isinstance(model.similarity, nn.CosineSimilarity):
                scores = model.similarity(
                    F.normalize(user_embedding, dim=-1),
                    F.normalize(candidate_embeddings, dim=-1),
                )
            else:
                scores = model.similarity(user_embedding, candidate_embeddings)

            topk = torch.topk(scores.squeeze(0), k=min(max_k, len(candidate_list)))
            predicted_indices = topk.indices.cpu().tolist()
            per_user_predictions[int(user_idx)] = [candidate_list[i] for i in predicted_indices]

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


def run_training(config: Mapping[str, Any]) -> None:
    """
    Execute the end-to-end training workflow.

    Parameters
    ----------
    config:
        Nested mapping parsed from YAML or another configuration source.
    """
    experiment_cfg = config.get("experiment", {})
    if "seed" in experiment_cfg:
        _seed_everything(int(experiment_cfg["seed"]))

    data_config = config.get("data", {})
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

    train_interactions, val_interactions = _split_train_validation(
        training_dataset.interactions
    )
    logger.info(
        "Split interactions | train={} validation={}",
        len(train_interactions),
        len(val_interactions),
    )

    training_config = config.get("training", {})
    batch_size = int(training_config.get("batch_size", 512))
    num_epochs = int(training_config.get("num_epochs", 10))
    learning_rate = float(training_config.get("learning_rate", 1e-3))
    weight_decay = float(training_config.get("weight_decay", 0.0))
    negatives_per_positive = int(training_config.get("negatives_per_positive", 5))

    model_config = config.get("model", {})
    device_name = model_config.get("device", "cpu")
    device = torch.device(device_name)

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

    model = TwoTowerModel(
        user_encoder=user_encoder,
        item_encoder=item_encoder,
        similarity=similarity_module,
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
        return

    criterion = nn.BCEWithLogitsLoss()

    if train_interactions.empty:
        logger.warning("No training interactions available; exiting early.")
        return

    train_loader = _build_dataloader(train_interactions, batch_size=batch_size)

    for epoch in range(1, num_epochs + 1):
        avg_loss = _train_one_epoch(
            model,
            train_loader,
            optimizers=optimizers,
            criterion=criterion,
            negatives_per_positive=negatives_per_positive,
            num_items=len(training_dataset.item_mapping),
            user_positive_items=training_dataset.user_positive_items,
            user_features=user_feature_tensor,
            item_features=item_feature_tensor,
            device=device,
        )
        logger.info("Epoch {:03d}/{:03d} | loss={:.4f}", epoch, num_epochs, avg_loss)

    eval_cfg = config.get("evaluation", {})
    metrics_k = eval_cfg.get("metrics_k", [10])
    if isinstance(metrics_k, int):
        metrics_k = [metrics_k]
    candidate_samples = int(eval_cfg.get("candidate_samples", 500))
    rng = np.random.default_rng(int(experiment_cfg.get("seed", 0)))
    num_items = len(training_dataset.item_mapping)

    train_positive_map = {
        int(user_idx): set(map(int, group["item_idx"].tolist()))
        for user_idx, group in train_interactions.groupby("user_idx")
    }

    per_user_predictions, per_user_ground_truth = _evaluate_model(
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
    )

    metrics_summary = compute_ranking_metrics(
        per_user_predictions, per_user_ground_truth, metrics_k
    )
    for k in metrics_k:
        logger.info(
            "Validation metrics @{} | recall={:.4f} precision={:.4f} ndcg={:.4f} hit_rate={:.4f} map={:.4f}",
            k,
            metrics_summary.recall[k],
            metrics_summary.precision[k],
            metrics_summary.ndcg[k],
            metrics_summary.hit_rate[k],
            metrics_summary.map[k],
        )

    diag_cfg = config.get("diagnostics", {})
    item_sample_size = int(diag_cfg.get("item_sample_size", 500))
    user_sample_size = int(diag_cfg.get("user_sample_size", 5000))
    neighbor_k = int(diag_cfg.get("neighbor_k", 10))
    report_path = Path(diag_cfg.get("report_path", "artifacts/reports/recommendation_report.md"))

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
        item_sample_embeddings = model.item_encoder(item_inputs).detach().cpu()
        item_sample_frame = items_df.loc[item_sample_indices.cpu().numpy()].reset_index(drop=True)
    else:
        item_sample_embeddings = torch.empty((0, model.item_encoder.output_dim), dtype=torch.float32)
        item_sample_frame = items_df.iloc[0:0]

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
        user_sample_embeddings = model.user_encoder(user_inputs).detach().cpu()
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

    rec_cfg = config.get("recommendations", {})
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

    _write_recommendation_report(
        report_path,
        metrics_summary=metrics_summary,
        embedding_stats=embedding_stats,
        recommendations=recommendation_samples,
    )
