"""
Adaptive mimic mechanism used by the DAT training objective.

The module owns per-user and per-item augmentation tables. During training
these vectors are nudged towards the opposite tower's representations via a
mimic loss where the target tower is stop-grad'ed. The augmented vectors are
then added back to the base tower outputs to form the final embeddings that
participate in retrieval.
"""

from __future__ import annotations

from typing import Optional, Tuple

import torch
import torch.nn.functional as F
from torch import nn


class AdaptiveMimicMechanism(nn.Module):
    """Adaptive mimic implementation with trainable augmentation tables."""

    def __init__(
        self,
        *,
        num_users: int,
        num_items: int,
        embedding_dim: int,
        init_std: float = 0.02,
    ) -> None:
        super().__init__()
        if num_users <= 0 or num_items <= 0:
            raise ValueError("num_users and num_items must be positive.")
        self.embedding_dim = int(embedding_dim)
        self.user_augmented = nn.Embedding(num_users, self.embedding_dim)
        self.item_augmented = nn.Embedding(num_items, self.embedding_dim)
        nn.init.normal_(self.user_augmented.weight, mean=0.0, std=init_std)
        nn.init.normal_(self.item_augmented.weight, mean=0.0, std=init_std)

    def forward(
        self,
        *,
        user_indices: torch.Tensor | None,
        item_indices: torch.Tensor | None,
        user_embedding: torch.Tensor,
        item_embedding: torch.Tensor,
    ) -> Tuple[
        torch.Tensor,
        torch.Tensor,
        torch.Tensor | None,
        torch.Tensor | None,
    ]:
        """
        Augment tower outputs and compute mimic losses for positive pairs.
        """
        if user_indices is None or item_indices is None:
            raise ValueError("user_indices and item_indices are required for mimic.")

        augmented_user, user_aug = self._apply_aug(
            self.user_augmented, user_indices, user_embedding
        )
        augmented_item, item_aug = self._apply_aug(
            self.item_augmented, item_indices, item_embedding
        )

        mimic_user_loss = F.mse_loss(user_aug, item_embedding.detach())
        mimic_item_loss = F.mse_loss(item_aug, user_embedding.detach())
        return augmented_user, augmented_item, mimic_user_loss, mimic_item_loss

    def augment_users(
        self, indices: Optional[torch.Tensor], base_embedding: torch.Tensor
    ) -> torch.Tensor:
        """Add the user-side augmentation to a base embedding."""
        if indices is None:
            return base_embedding
        augmented, _ = self._apply_aug(self.user_augmented, indices, base_embedding)
        return augmented

    def augment_items(
        self, indices: Optional[torch.Tensor], base_embedding: torch.Tensor
    ) -> torch.Tensor:
        """Add the item-side augmentation to a base embedding."""
        if indices is None:
            return base_embedding
        augmented, _ = self._apply_aug(self.item_augmented, indices, base_embedding)
        return augmented

    def _apply_aug(
        self,
        table: nn.Embedding,
        indices: torch.Tensor,
        reference: torch.Tensor,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        aug = self._gather_and_reshape(table, indices, reference)
        return reference + aug, aug

    @staticmethod
    def _gather_and_reshape(
        table: nn.Embedding, indices: torch.Tensor, reference: torch.Tensor
    ) -> torch.Tensor:
        if indices.dtype != torch.long:
            raise ValueError("Adaptive mimic indices must be torch.long tensors.")
        flat = indices.reshape(-1)
        gathered = table(flat)
        return gathered.reshape(reference.shape)
