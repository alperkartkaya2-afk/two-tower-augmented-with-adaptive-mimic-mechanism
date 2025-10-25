"""
Core two-tower module skeleton.

Extending this class with task-specific encoder stacks keeps the public API
stable while enabling modular experimentation (e.g., swapping Transformers,
bag-of-words encoders, or metadata MLPs).
"""

from __future__ import annotations

from typing import Any

import torch
from torch import nn

from .adaptive_mimic import AdaptiveMimicMechanism


class TwoTowerModel(nn.Module):
    """
    Minimal two-tower template with separate user and item encoders.

    The default implementation defers actual encoder construction to subclasses
    or factory helpers so this file can remain a thin orchestrator.
    """

    def __init__(
        self,
        user_encoder: nn.Module,
        item_encoder: nn.Module,
        similarity: nn.Module | None = None,
        adaptive_mimic: AdaptiveMimicMechanism | None = None,
    ) -> None:
        super().__init__()
        self.user_encoder = user_encoder
        self.item_encoder = item_encoder
        self.similarity = similarity or nn.CosineSimilarity(dim=-1)
        self.adaptive_mimic = adaptive_mimic

    def forward(
        self, user_inputs: Any, item_inputs: Any, *, return_embeddings: bool = False
    ) -> dict[str, torch.Tensor]:
        """
        Compute embeddings for both towers and optionally return similarity scores.

        Parameters
        ----------
        user_inputs, item_inputs:
            Model-specific inputs (tensors, dicts, etc.) that the encoders expect.
        return_embeddings:
            When True, the output dictionary includes the raw tower embeddings.
        """
        user_embedding = self.user_encoder(user_inputs)
        item_embedding = self.item_encoder(item_inputs)

        mimic_user_loss: torch.Tensor | None = None
        mimic_item_loss: torch.Tensor | None = None
        if self.adaptive_mimic is not None:
            user_indices = _extract_indices(user_inputs)
            item_indices = _extract_indices(item_inputs)
            (
                user_embedding,
                item_embedding,
                mimic_user_loss,
                mimic_item_loss,
            ) = self.adaptive_mimic(
                user_indices=user_indices,
                item_indices=item_indices,
                user_embedding=user_embedding,
                item_embedding=item_embedding,
            )

        outputs: dict[str, torch.Tensor] = {}
        if return_embeddings:
            outputs["user_embedding"] = user_embedding
            outputs["item_embedding"] = item_embedding
        if mimic_user_loss is not None:
            outputs["mimic_user_loss"] = mimic_user_loss
        if mimic_item_loss is not None:
            outputs["mimic_item_loss"] = mimic_item_loss

        outputs["score"] = self.similarity(user_embedding, item_embedding)
        return outputs


def _extract_indices(inputs: Any) -> torch.Tensor | None:
    if inputs is None:
        return None
    if isinstance(inputs, torch.Tensor):
        return inputs
    if isinstance(inputs, dict):
        candidate = inputs.get("indices")
        if isinstance(candidate, torch.Tensor):
            return candidate
    return None
