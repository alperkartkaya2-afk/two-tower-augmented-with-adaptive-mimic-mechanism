"""Negative sampling utilities."""

from __future__ import annotations

import random
from typing import Mapping, Set

import torch


def sample_negative_items(
    user_indices: torch.Tensor,
    *,
    num_items: int,
    positives: Mapping[int, Set[int]],
    num_negatives: int,
    device: torch.device,
) -> torch.Tensor:
    """
    Sample negative item indices for each user in the batch.

    Parameters
    ----------
    user_indices:
        Tensor containing user indices for the current batch.
    num_items:
        Total number of distinct item indices available.
    positives:
        Mapping of user index -> set of known positive item indices.
    num_negatives:
        Number of negatives to sample per user.
    device:
        Device on which the returned tensor should live.
    """
    if num_negatives <= 0:
        raise ValueError("num_negatives must be greater than zero.")

    batch_size = user_indices.shape[0]
    negatives = torch.empty(
        (batch_size, num_negatives), dtype=torch.long, device=device
    )

    all_items = list(range(num_items))

    for row, user_idx in enumerate(user_indices.tolist()):
        positives_for_user = positives.get(int(user_idx), set())
        available_items = [item for item in all_items if item not in positives_for_user]

        if len(available_items) < num_negatives:
            raise RuntimeError(
                f"Unable to sample {num_negatives} negatives for user {user_idx}; "
                "not enough unseen items."
            )

        sampled = random.sample(available_items, num_negatives)
        negatives[row] = torch.as_tensor(sampled, dtype=torch.long, device=device)

    return negatives
