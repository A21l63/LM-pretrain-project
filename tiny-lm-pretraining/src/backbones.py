"""Backbone modules used inside the shared LanguageModel interface."""

from __future__ import annotations

import torch
from torch import nn


class GRUBackbone(nn.Module):
    """GRU backbone skeleton.

    TODO(student): use nn.GRU with batch_first=True and return a tensor with
    shape [B, T, D]. Add a projection if hidden_size != d_model.
    """

    def __init__(self, d_model: int, hidden_size: int, num_layers: int = 1, dropout: float = 0.0) -> None:
        super().__init__()
        self.d_model = d_model
        self.hidden_size = hidden_size
        self.num_layers = num_layers
        self.dropout = dropout
        self.gru = nn.GRU(input_size=d_model, hidden_size=hidden_size, num_layers=num_layers, batch_first=True,
                          dropout=dropout if num_layers > 1 else 0.0)
        self.proj = nn.Linear(hidden_size, d_model) if hidden_size != d_model else nn.Identity()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        out, _ = self.gru(x)
        return self.proj(out)


class TransformerBlock(nn.Module):
    """One decoder-only Transformer block.

    TODO(student): implement pre-norm causal self-attention, residual
    connections, MLP, and dropout.
    """

    def __init__(self, d_model: int, num_heads: int, dropout: float = 0.1) -> None:
        super().__init__()
        self.ln_1 = nn.LayerNorm(d_model)
        self.attn = nn.MultiheadAttention(
            embed_dim=d_model,
            num_heads=num_heads,
            dropout=dropout,
            batch_first=True,
        )
        self.ln_2 = nn.LayerNorm(d_model)
        self.mlp = nn.Sequential(
            nn.Linear(d_model, 4 * d_model),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(4 * d_model, d_model),
            nn.Dropout(dropout),
        )
        self.dropout = nn.Dropout(dropout)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        normed = self.ln_1(x)
        T = normed.size(1)
        causal_mask = torch.triu(
            torch.ones(T, T, device=x.device, dtype=torch.bool),
            diagonal=1,
        )
        attn_out, _ = self.attn(normed, normed, normed, attn_mask=causal_mask)
        x = x + self.dropout(attn_out)

        normed = self.ln_2(x)
        mlp_out = self.mlp(normed)
        x = x + self.dropout(mlp_out)

        return x


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
        self.position_embedding = nn.Embedding(block_size, d_model)
        self.blocks = nn.ModuleList(
            [TransformerBlock(d_model, num_heads, dropout) for _ in range(num_layers)]
        )
        self.final_ln = nn.LayerNorm(d_model)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        B, T, _ = x.shape
        pos = torch.arange(T, device=x.device).unsqueeze(0)
        pos_emb = self.position_embedding(pos)
        x = x + pos_emb

        for block in self.blocks:
            x = block(x)

        return self.final_ln(x)
