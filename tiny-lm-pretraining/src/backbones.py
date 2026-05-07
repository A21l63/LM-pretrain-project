"""Backbone modules used inside the shared LanguageModel interface."""

from __future__ import annotations

import torch
from torch import nn


class GRUBackbone(nn.Module):
    """GRU backbone skeleton.

    TODO(student): use nn.GRU with batch_first=True and return a tensor with
    shape [B, T, D]. Add a projection if hidden_size != d_model.
    """

    def __init__(
        self,
        d_model: int,
        hidden_size: int,
        num_layers: int = 1,
        dropout: float = 0.0,
    ) -> None:
        super().__init__()
        self.d_model = d_model
        self.hidden_size = hidden_size
        self.num_layers = num_layers
        self.dropout = dropout
        self.gru = ...
        self.proj = ...

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        ...


class TransformerBlock(nn.Module):
    """One decoder-only Transformer block.

    TODO(student): implement pre-norm causal self-attention, residual
    connections, MLP, and dropout.
    """

    def __init__(self, d_model: int, num_heads: int, dropout: float = 0.1) -> None:
        super().__init__()
        self.ln_1 = ...
        self.attn = ...
        self.ln_2 = ...
        self.mlp = ...
        self.dropout = ...

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        ...


class TransformerBackbone(nn.Module):
    """Stack of decoder-only Transformer blocks.

    TODO(student): add positional embeddings and ensure attention is causal.
    """

    def __init__(
        self,
        block_size: int,
        d_model: int,
        num_layers: int = 2,
        num_heads: int = 4,
        dropout: float = 0.1,
    ) -> None:
        super().__init__()
        self.block_size = block_size
        self.position_embedding = ...
        self.blocks = ...
        self.final_ln = ...

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        ...
