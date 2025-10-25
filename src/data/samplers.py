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

    Sampling is performed in torch so the operation remains vectorised and avoids
    Python-level loops over candidate pools.
    """
    if num_negatives <= 0:
        raise ValueError("num_negatives must be greater than zero.")
    if num_items <= 1:
        raise ValueError("num_items must be greater than one.")

    batch_size = user_indices.shape[0]
    negatives = torch.empty(
        (batch_size, num_negatives), dtype=torch.long, device=device
    )
    cached_positive_tensors: dict[int, torch.Tensor] = {}

    for row, user_idx in enumerate(user_indices.tolist()):
        user_idx_int = int(user_idx)
        positives_for_user = positives.get(user_idx_int, set())
        if len(positives_for_user) >= num_items:
            raise RuntimeError(
                f"User {user_idx_int} interacted with all items; cannot sample negatives."
            )

        if positives_for_user:
            pos_tensor = cached_positive_tensors.get(user_idx_int)
            if pos_tensor is None:
                pos_tensor = torch.tensor(
                    sorted(positives_for_user),
                    device=device,
                    dtype=torch.long,
                )
                cached_positive_tensors[user_idx_int] = pos_tensor
        else:
            pos_tensor = None

        samples = torch.randint(
            low=0,
            high=num_items,
            size=(num_negatives,),
            device=device,
            dtype=torch.long,
        )

        if pos_tensor is not None:
            invalid = torch.isin(samples, pos_tensor)
            attempts = 0
            while invalid.any():
                resample = torch.randint(
                    low=0,
                    high=num_items,
                    size=(int(invalid.sum().item()),),
                    device=device,
                    dtype=torch.long,
                )
                samples[invalid] = resample
                invalid = torch.isin(samples, pos_tensor)
                attempts += 1
                if attempts > 10:
                    raise RuntimeError(
                        "Exceeded resampling attempts while drawing negatives."
                    )

        negatives[row] = samples

    return negatives
