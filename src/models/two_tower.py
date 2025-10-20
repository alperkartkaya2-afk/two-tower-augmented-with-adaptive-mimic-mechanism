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
    ) -> None:
        super().__init__()
        self.user_encoder = user_encoder
        self.item_encoder = item_encoder
        self.similarity = similarity or nn.CosineSimilarity(dim=-1)

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

        outputs: dict[str, torch.Tensor] = {}
        if return_embeddings:
            outputs["user_embedding"] = user_embedding
            outputs["item_embedding"] = item_embedding

        outputs["score"] = self.similarity(user_embedding, item_embedding)
        return outputs

