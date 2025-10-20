"""
Dataset wrappers for turning pandas interactions into PyTorch-friendly tensors.
"""

from __future__ import annotations

import pandas as pd
import torch
from torch.utils.data import Dataset


class InteractionDataset(Dataset):
    """
    Lightweight dataset containing user/item index pairs.

    Parameters
    ----------
    interactions:
        DataFrame with at least `user_idx` and `item_idx` columns.
    """

    def __init__(
        self,
        interactions: pd.DataFrame,
        *,
        user_col: str = "user_idx",
        item_col: str = "item_idx",
    ) -> None:
        if user_col not in interactions.columns or item_col not in interactions.columns:
            raise ValueError(
                f"Interactions must contain '{user_col}' and '{item_col}' columns."
            )

        self._users = torch.as_tensor(
            interactions[user_col].to_numpy(), dtype=torch.long
        )
        self._items = torch.as_tensor(
            interactions[item_col].to_numpy(), dtype=torch.long
        )

    def __len__(self) -> int:
        return self._users.shape[0]

    def __getitem__(self, idx: int) -> tuple[torch.Tensor, torch.Tensor]:
        return self._users[idx], self._items[idx]

