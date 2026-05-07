"""Tokenizer interfaces for the tiny LM project."""

from __future__ import annotations


class BaseTokenizer:
    """Common tokenizer interface used by all experiments."""

    def train(self, texts: list[str]) -> None:
        raise NotImplementedError

    def encode(self, text: str) -> list[int]:
        raise NotImplementedError

    def decode(self, ids: list[int]) -> str:
        raise NotImplementedError

    @property
    def vocab_size(self) -> int:
        raise NotImplementedError


class CharTokenizer(BaseTokenizer):
    """Character-level tokenizer skeleton.

    TODO(student): build a vocabulary from characters in `texts`, reserve ids
    for special tokens if you use them, and implement encode/decode.
    """

    def __init__(self) -> None:
        self.stoi: dict[str, int] = {}
        self.itos: dict[int, str] = {}

    def train(self, texts: list[str]) -> None:
        ...

    def encode(self, text: str) -> list[int]:
        ...

    def decode(self, ids: list[int]) -> str:
        ...

    @property
    def vocab_size(self) -> int:
        ...


class HFTokenizerWrapper(BaseTokenizer):
    """Optional wrapper around a pretrained Hugging Face tokenizer.

    TODO(student): instantiate AutoTokenizer and map its encode/decode methods
    to the BaseTokenizer interface. Do not train a large tokenizer here.
    """

    def __init__(self, name: str = "gpt2") -> None:
        self.name = name
        self.tokenizer = None

    def train(self, texts: list[str]) -> None:
        ...

    def encode(self, text: str) -> list[int]:
        ...

    def decode(self, ids: list[int]) -> str:
        ...

    @property
    def vocab_size(self) -> int:
        ...
