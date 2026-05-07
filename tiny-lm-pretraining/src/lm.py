"""Shared language model interface for GRU and Transformer experiments."""

from __future__ import annotations

import torch
import torch.nn.functional as F
from torch import nn

from src.backbones import GRUBackbone, TransformerBackbone


class BaseLanguageModel(nn.Module):
    """Base class for causal language models.

    The neural network operates on token ids. Tokenization stays conceptually
    outside the model, even if a tokenizer is stored for convenience.
    """

    def __init__(
        self,
        vocab_size: int,
        block_size: int,
        d_model: int,
        tokenizer=None,
    ) -> None:
        super().__init__()
        self.vocab_size = vocab_size
        self.block_size = block_size
        self.d_model = d_model
        self.tokenizer = tokenizer
        self.token_embedding = ...
        self.backbone = ...
        self.lm_head = ...

    def build_backbone(self) -> nn.Module:
        """Subclasses return GRUBackbone or TransformerBackbone here."""
        raise NotImplementedError

    def forward(
        self, input_ids: torch.Tensor, labels: torch.Tensor | None = None
    ) -> tuple[torch.Tensor, torch.Tensor | None]:
        """Return logits and optional next-token prediction loss.

        Expected shapes:
        input_ids: [B, T]
        embeddings: [B, T, D]
        backbone output: [B, T, D]
        logits: [B, T, V]
        labels: [B, T]
        loss input logits: [B*T, V]
        loss input labels: [B*T]
        """
        ...

    @torch.no_grad()
    def generate(
        self,
        input_ids: torch.Tensor,
        max_new_tokens: int,
        temperature: float = 1.0,
        top_k: int | None = None,
    ) -> torch.Tensor:
        """Generate token ids by repeatedly sampling the next token."""
        ...

    @torch.no_grad()
    def generate_text(
        self,
        prompt: str,
        tokenizer=None,
        max_new_tokens: int = 100,
        temperature: float = 1.0,
        top_k: int | None = None,
    ) -> str:
        """Optional convenience wrapper around generate()."""
        ...

    def count_parameters(self) -> int:
        """Return number of trainable parameters."""
        ...


class GRULanguageModel(BaseLanguageModel):
    """Language model whose backbone is a GRU."""

    def __init__(
        self,
        vocab_size: int,
        block_size: int,
        d_model: int,
        hidden_size: int = 128,
        num_layers: int = 1,
        dropout: float = 0.1,
        tokenizer=None,
    ) -> None:
        self.hidden_size = hidden_size
        self.num_layers = num_layers
        self.dropout = dropout
        super().__init__(vocab_size, block_size, d_model, tokenizer=tokenizer)

    def build_backbone(self) -> nn.Module:
        ...


class TransformerLanguageModel(BaseLanguageModel):
    """Language model whose backbone is a decoder-only Transformer."""

    def __init__(
        self,
        vocab_size: int,
        block_size: int,
        d_model: int,
        num_layers: int = 2,
        num_heads: int = 4,
        dropout: float = 0.1,
        tokenizer=None,
    ) -> None:
        self.num_layers = num_layers
        self.num_heads = num_heads
        self.dropout = dropout
        super().__init__(vocab_size, block_size, d_model, tokenizer=tokenizer)

    def build_backbone(self) -> nn.Module:
        ...
