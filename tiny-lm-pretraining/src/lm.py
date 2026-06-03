from __future__ import annotations
from wsgiref.util import request_uri
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
        self.token_embedding = nn.Embedding(vocab_size, d_model)
        self.backbone = self.build_backbone()
        self.lm_head = nn.Linear(d_model, vocab_size)

    def build_backbone(self) -> nn.Module:
        """Subclasses return GRUBackbone or TransformerBackbone here."""
        raise NotImplementedError

    def forward(
        self, input_ids: torch.Tensor, labels: torch.Tensor | None = None
    ) -> tuple[torch.Tensor, torch.Tensor | None]:

        x = self.token_embedding(input_ids)
        x = self.backbone(x)
        logits = self.lm_head(x)
        loss = None
        if labels is not None: #если есть ответы
            loss = F.cross_entropy(logits.view(-1,self.vocab_size), labels.view(-1))
            #применяет softmax (exp(x_j)/сумму exp(x_j)для каждого слова считает loss=-ln(вер правильного ответа)

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
        return logits, loss

    @torch.no_grad()
    def generate(
        self,
        input_ids: torch.Tensor,
        max_new_tokens: int,
        temperature: float = 1.0,
        top_k: int | None = None, #к примеру передаётся 50, значит модель оставит лишь 50 самых вероятных слов
    ) -> torch.Tensor:
        """Generate token ids by repeatedly sampling the next token."""
        for _ in range(max_new_tokens):  # на каждом шаге добавляем одно новое слово
            input_cond = input_ids[:, -self.block_size:] #обрезаем текст чтобы влез в память
            logits , _ = self.forward(input_cond)
            logits = logits[:, -1, :] / temperature
            if top_k is not None:
                v, _ = torch.topk(logits, min(top_k, logits.size(-1)))
                logits[logits < v[:, [-1]]] = -float('Inf')
            probs = F.softmax(logits, dim=-1) #dim=-1 означает по какому измерению считать, на вход у нас в данном случае [batch, vocab_size], считаем по vocab
            next_id = torch.multinomial(probs, num_samples=1)
            input_ids = torch.cat((input_ids, next_id), dim=-1)
        return input_ids

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
        tok = tokenizer if tokenizer is not None else self.tokenizer
        input_ids = tok.encode(prompt, return_tensors="pt").to(self.lm_head.weight.device)
        output_ids = self.generate(
            input_ids,
            max_new_tokens=max_new_tokens,
            temperature=temperature,
            top_k=top_k
        )
        return tok.decode(output_ids[0], skip_special_tokens=True)

    def count_parameters(self) -> int:
        """Return number of trainable parameters."""
        return sum(p.numel() for p in self.parameters() if p.requires_grad) #self parameters обходит все слои, embeddiing, backbone, head. p_req... этот параметр говорить нам, что мы считаем незаблокированные вес
    #p.numel - количество эл-тов (чисел) если 1000 векторов по 1000 эл, то вернёт 1_000_000


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
        return GRUBackbone(
            d_model = self.d_model,
            hidden_size = self.hidden_size,
            num_layers = self.num_layers,
            dropout = self.dropout
        )


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
        return TransformerBackbone(
            d_model = self.d_model,
            num_heads = self.num_heads,
            num_layers = self.num_layers,
            dropout = self.dropout,
            block_size = self.block_size
        )
