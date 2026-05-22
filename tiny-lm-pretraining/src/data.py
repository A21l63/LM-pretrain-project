"""Dataset utilities for next-token prediction."""

from __future__ import annotations

import torch
from torch.utils.data import Dataset


class LMDataset(Dataset):
    """A dataset that returns shifted input/target token blocks.

    For token ids [t0, t1, t2, t3, t4] and block_size=4:
    x = [t0, t1, t2, t3]
    y = [t1, t2, t3, t4]
    """

    def __init__(self, token_ids: list[int], block_size: int):
        self.token_ids = token_ids
        self.block_size = block_size

    def __len__(self) -> int:
        return max(0, len(self.token_ids) - self.block_size)

    def __getitem__(self, idx: int) -> tuple[torch.Tensor, torch.Tensor]:
        x = self.token_ids[idx: idx + self.block_size]
        y = self.token_ids[idx + 1: idx + self.block_size + 1]
        return torch.tensor(x, dtype=torch.long), torch.tensor(y, dtype=torch.long)


def split_token_ids(
    token_ids: list[int], validation_fraction: float = 0.1
) -> tuple[list[int], list[int]]:
    """Split one token stream into train and validation parts."""
    split_idx = int(len(token_ids) * (1 - validation_fraction))
    train_ids = token_ids[:split_idx]
    val_ids = token_ids[split_idx:]
    return train_ids, val_ids