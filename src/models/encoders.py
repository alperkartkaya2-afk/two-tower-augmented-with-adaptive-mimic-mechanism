"""
Reusable encoder building blocks for the two-tower architecture.

Extends beyond pure ID embeddings to include feature-driven towers and an
adaptive mimic module that allows item/user representations to borrow strength
from side information.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Iterable, Mapping

import torch
from torch import nn


def _init_embedding(
    embedding: nn.Embedding, init_config: Mapping[str, Any] | None = None
) -> None:
    init_config = init_config or {"type": "normal", "std": 0.02}
    init_type = init_config.get("type", "normal").lower()

    if init_type == "normal":
        std = float(init_config.get("std", 0.02))
        nn.init.normal_(embedding.weight, mean=0.0, std=std)
    elif init_type == "uniform":
        bound = float(init_config.get("bound", 0.1))
        nn.init.uniform_(embedding.weight, -bound, bound)
    elif init_type == "xavier_normal":
        nn.init.xavier_normal_(embedding.weight)
    elif init_type == "xavier_uniform":
        nn.init.xavier_uniform_(embedding.weight)
    else:  # pragma: no cover - defensive branch
        raise ValueError(f"Unsupported embedding init type: {init_type}")


def build_id_embedding(
    config: Mapping[str, Any],
    *,
    num_embeddings: int,
    device: torch.device | None = None,
) -> nn.Embedding:
    params = config.get("params", {})
    embedding_dim = int(params.get("embedding_dim", 64))
    padding_idx = params.get("padding_idx")
    max_norm = params.get("max_norm")
    sparse = bool(params.get("sparse", False))

    if sparse and max_norm is not None:
        raise ValueError("max_norm is not supported when using sparse embeddings.")

    embedding = nn.Embedding(
        num_embeddings=num_embeddings,
        embedding_dim=embedding_dim,
        padding_idx=padding_idx,
        max_norm=max_norm,
        sparse=sparse,
    )
    _init_embedding(embedding, config.get("init"))

    if device is not None:
        embedding = embedding.to(device)
    return embedding


def _get_activation(name: str) -> nn.Module:
    key = name.lower()
    if key == "relu":
        return nn.ReLU()
    if key == "gelu":
        return nn.GELU()
    if key == "tanh":
        return nn.Tanh()
    if key == "selu":
        return nn.SELU()
    raise ValueError(f"Unsupported activation '{name}'")


class FeatureEncoderWrapper(nn.Module):
    """Wraps a projection network while exposing its output dimension."""

    def __init__(self, network: nn.Module, output_dim: int) -> None:
        super().__init__()
        self.network = network
        self.output_dim = output_dim

    def forward(self, inputs: torch.Tensor) -> torch.Tensor:
        return self.network(inputs)


@dataclass(frozen=True)
class FeatureEncoderConfig:
    type: str = "linear"
    output_dim: int | None = None
    hidden_dims: Iterable[int] | None = None
    activation: str = "relu"
    dropout: float = 0.0


def build_feature_encoder(
    config: Mapping[str, Any] | None,
    *,
    input_dim: int,
    fallback_output_dim: int,
) -> FeatureEncoderWrapper | None:
    if input_dim == 0:
        return None

    cfg = FeatureEncoderConfig(**(config or {}))
    output_dim = int(cfg.output_dim or fallback_output_dim)

    if cfg.type == "identity":
        if input_dim != output_dim:
            raise ValueError(
                "Identity feature encoder requires input_dim == output_dim."
            )
        return FeatureEncoderWrapper(nn.Identity(), output_dim)

    if cfg.type == "linear":
        layer = nn.Linear(input_dim, output_dim)
        nn.init.xavier_uniform_(layer.weight)
        return FeatureEncoderWrapper(layer, output_dim)

    if cfg.type == "mlp":
        hidden_dims = [int(h) for h in (cfg.hidden_dims or [])]
        layers: list[nn.Module] = []
        prev_dim = input_dim
        activation = _get_activation(cfg.activation)

        for hidden_dim in hidden_dims:
            linear = nn.Linear(prev_dim, hidden_dim)
            nn.init.xavier_uniform_(linear.weight)
            layers.extend([linear, activation])
            if cfg.dropout:
                layers.append(nn.Dropout(p=cfg.dropout))
            prev_dim = hidden_dim

        final_linear = nn.Linear(prev_dim, output_dim)
        nn.init.xavier_uniform_(final_linear.weight)
        layers.append(final_linear)

        return FeatureEncoderWrapper(nn.Sequential(*layers), output_dim)

    raise ValueError(f"Unsupported feature encoder type: {cfg.type}")


class AdaptiveMimicModule(nn.Module):
    """
    Blends ID embeddings with feature-derived representations using a gated mixer.
    """

    def __init__(self, dim: int, hidden_dim: int | None = None) -> None:
        super().__init__()
        hidden = hidden_dim or dim
        self.gate_network = nn.Sequential(
            nn.Linear(dim * 2, hidden),
            nn.ReLU(),
            nn.Linear(hidden, dim),
            nn.Sigmoid(),
        )

    def forward(
        self, id_repr: torch.Tensor, feature_repr: torch.Tensor
    ) -> torch.Tensor:
        gate = self.gate_network(torch.cat([id_repr, feature_repr], dim=-1))
        return gate * id_repr + (1.0 - gate) * feature_repr


class TowerEncoder(nn.Module):
    """
    Combines ID embeddings with optional feature encoders and mimic mechanisms.
    """

    def __init__(
        self,
        *,
        embedding: nn.Embedding,
        feature_encoder: FeatureEncoderWrapper | None,
        fusion: str,
        output_dim: int | None,
        adaptive_mimic: AdaptiveMimicModule | None,
    ) -> None:
        super().__init__()
        self.embedding = embedding
        self.feature_encoder = feature_encoder
        self.fusion = fusion
        self.adaptive_mimic = adaptive_mimic
        self.num_embeddings = embedding.num_embeddings
        self.id_dim = embedding.embedding_dim
        self.output_dim = self.id_dim

        if fusion not in {"identity", "sum", "concat", "adaptive_mimic"}:
            raise ValueError(f"Unsupported fusion strategy: {fusion}")

        if feature_encoder is None:
            self.fusion = "identity"

        if self.fusion == "concat":
            target_dim = int(output_dim or (self.id_dim + feature_encoder.output_dim))
            self.projection = nn.Linear(
                self.id_dim + feature_encoder.output_dim, target_dim
            )
            nn.init.xavier_uniform_(self.projection.weight)
            self.output_dim = target_dim
        else:
            self.output_dim = self.id_dim

    def forward(self, inputs: Mapping[str, torch.Tensor]) -> torch.Tensor:
        indices = inputs["indices"]
        id_repr = self.embedding(indices)

        if self.fusion == "identity" or self.feature_encoder is None:
            return id_repr

        features = inputs.get("features")
        if features is None:
            # Fallback to id-only behaviour when features are unavailable.
            return id_repr

        feature_repr = self.feature_encoder(features)

        if self.fusion == "sum":
            if feature_repr.shape[-1] != id_repr.shape[-1]:
                raise ValueError(
                    "Feature encoder output dimension must match id embedding dimension for 'sum' fusion."
                )
            return id_repr + feature_repr

        if self.fusion == "concat":
            combined = torch.cat([id_repr, feature_repr], dim=-1)
            return self.projection(combined)

        if self.fusion == "adaptive_mimic":
            if self.adaptive_mimic is None:
                raise ValueError("Adaptive mimic fusion requires a mimic module.")
            if feature_repr.shape[-1] != id_repr.shape[-1]:
                raise ValueError(
                    "Adaptive mimic requires feature encoder output to match id embedding dimension."
                )
            return self.adaptive_mimic(id_repr, feature_repr)

        return id_repr


def build_tower_encoder(
    config: Mapping[str, Any] | None,
    *,
    num_embeddings: int,
    feature_dim: int,
    device: torch.device | None = None,
) -> TowerEncoder:
    cfg = config or {}
    encoder_type = cfg.get("type", "tower").lower()

    if encoder_type not in {"tower", "embedding"}:
        raise ValueError(f"Unsupported encoder type: {encoder_type}")

    if encoder_type == "embedding":
        embedding_cfg = {
            "params": cfg.get("params", {}),
            "init": cfg.get("init"),
        }
        embedding = build_id_embedding(
            {"params": embedding_cfg["params"], "init": embedding_cfg["init"]},
            num_embeddings=num_embeddings,
            device=device,
        )
        return TowerEncoder(
            embedding=embedding,
            feature_encoder=None,
            fusion="identity",
            output_dim=None,
            adaptive_mimic=None,
        ).to(device)

    id_embedding_cfg = cfg.get("id_embedding", {})
    embedding = build_id_embedding(
        {"params": id_embedding_cfg.get("params", {}), "init": id_embedding_cfg.get("init")},
        num_embeddings=num_embeddings,
        device=device,
    )

    fusion = cfg.get(
        "fusion", "adaptive_mimic" if feature_dim > 0 else "identity"
    ).lower()
    output_dim = cfg.get("output_dim")

    feature_encoder = build_feature_encoder(
        cfg.get("feature_encoder"),
        input_dim=feature_dim,
        fallback_output_dim=embedding.embedding_dim,
    )

    # When fusion expects matching dimensions, force feature encoder output to match.
    if fusion in {"sum", "adaptive_mimic"} and feature_encoder is not None:
        if feature_encoder.output_dim != embedding.embedding_dim:
            raise ValueError(
                "Feature encoder output dimension must equal embedding dimension for 'sum' or 'adaptive_mimic' fusion."
            )

    mimic_cfg = cfg.get("adaptive_mimic", {}) if fusion == "adaptive_mimic" else None
    mimic_module = None
    if mimic_cfg is not None:
        mimic_module = AdaptiveMimicModule(
            dim=embedding.embedding_dim,
            hidden_dim=mimic_cfg.get("hidden_dim"),
        )

    tower = TowerEncoder(
        embedding=embedding,
        feature_encoder=feature_encoder,
        fusion=fusion,
        output_dim=output_dim,
        adaptive_mimic=mimic_module,
    )
    if device is not None:
        tower = tower.to(device)
    return tower
