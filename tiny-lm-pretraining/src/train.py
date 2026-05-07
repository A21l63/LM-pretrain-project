"""Training and validation loops for tiny language models."""

from __future__ import annotations

from pathlib import Path
from typing import Any

import torch
from torch.utils.data import DataLoader
from tqdm import tqdm


def get_device() -> torch.device:
    """Choose CUDA if available, otherwise CPU."""
    ...


def train_one_step(
    model: torch.nn.Module,
    batch: tuple[torch.Tensor, torch.Tensor],
    optimizer: torch.optim.Optimizer,
    device: torch.device,
    grad_clip: float | None = None,
) -> float:
    """Run one optimization step and return the loss value."""
    ...

@torch.no_grad()
def evaluate_loss(
    model: torch.nn.Module,
    dataloader: DataLoader,
    device: torch.device,
    max_batches: int | None = None,
) -> float:
    """Compute average validation loss."""
    ...


def train_model(
    model: torch.nn.Module,
    train_loader: DataLoader,
    val_loader: DataLoader | None,
    optimizer: torch.optim.Optimizer,
    max_steps: int,
    device: torch.device | None = None,
    eval_interval: int = 100,
    grad_clip: float | None = None,
    checkpoint_path: str | Path | None = None,
) -> list[dict[str, Any]]:
    """Train any model that implements model(input_ids, labels)."""
    ...
