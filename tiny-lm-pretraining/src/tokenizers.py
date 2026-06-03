"""Tokenizer interfaces for the tiny LM project."""

from __future__ import annotations

from transformers import AutoTokenizer


def get_gpt2_tokenizer():
    tokenizer = AutoTokenizer.from_pretrained("gpt2")
    tokenizer.pad_token = tokenizer.eos_token

    return tokenizer

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
    for special tokens if you use them, and implement encode/decode.
    """

    def __init__(self) -> None:
        self.stoi: dict[str, int] = {}
        self.itos: dict[int, str] = {}

    def train(self, texts: list[str]) -> None:
        unique_chars = set()
        for text in texts:
            unique_chars.update(text)
        sorted_chars = sorted(unique_chars)
        self.stoi = {char:idx for idx, char in enumerate(sorted_chars)}
        self.itos = {idx:char for idx, char in enumerate(sorted_chars)}


    def encode(self, text: str) -> list[int]:
        return [self.stoi[char] for char in text]

    def decode(self, ids: list[int]) -> str:
        return "".join(self.itos[id] for id in ids)

    @property
    def vocab_size(self) -> int:
        return len(self.itos)


class HFTokenizerWrapper(BaseTokenizer):
    """Optional wrapper around a pretrained Hugging Face tokenizer.

    Instantiates AutoTokenizer and maps its encode/decode methods
    to the BaseTokenizer interface.
    """

    def __init__(self, name: str = "gpt2") -> None:
        self.name = name
        self.tokenizer = AutoTokenizer.from_pretrained(name)

        if self.tokenizer.pad_token is None:
            self.tokenizer.pad_token = self.tokenizer.eos_token

    def train(self, texts: list[str]) -> None:
        pass

    def encode(self, text: str) -> list[int]:
        return self.tokenizer.encode(text)

    def decode(self, ids: list[int]) -> str:
        return self.tokenizer.decode(ids)

    @property
    def vocab_size(self) -> int:
        return self.tokenizer.vocab_size