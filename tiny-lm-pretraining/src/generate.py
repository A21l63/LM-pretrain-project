"""Text generation utilities."""

from __future__ import annotations

import torch


def top_k_filter(logits: torch.Tensor, top_k: int | None) -> torch.Tensor:
    """Keep only top-k logits before sampling."""
    ...


def generate_text(
    model: torch.nn.Module,
    tokenizer,
    prompt: str,
    max_new_tokens: int = 100,
    temperature: float = 1.0,
    top_k: int | None = None,
    device: torch.device | None = None,
) -> str:
    """Encode prompt, call model.generate, and decode generated ids."""
    ...
