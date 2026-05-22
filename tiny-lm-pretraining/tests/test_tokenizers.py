"""Tests students should make pass while implementing tokenizers."""

import pytest

from src.tokenizers import CharTokenizer


def test_char_tokenizer_roundtrip() -> None:
    tokenizer = CharTokenizer()
    tokenizer.train(["hello", "world"])
    text = "hello"
    assert tokenizer.decode(tokenizer.encode(text)) == text


def test_char_tokenizer_vocab_size() -> None:
    tokenizer = CharTokenizer()
    tokenizer.train(["ab", "bc"])
    assert tokenizer.vocab_size == 3
