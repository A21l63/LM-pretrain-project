"""Shape tests students should make pass after implementing models."""

import pytest

torch = pytest.importorskip("torch")

from src.lm import GRULanguageModel, TransformerLanguageModel


@pytest.mark.skip(reason="TODO(student): implement GRULanguageModel first")
def test_gru_lm_shapes() -> None:
    model = GRULanguageModel(vocab_size=10, block_size=8, d_model=16)
    input_ids = torch.randint(0, 10, (2, 8))
    logits, loss = model(input_ids, labels=input_ids)
    assert logits.shape == (2, 8, 10)
    assert loss is not None


@pytest.mark.skip(reason="TODO(student): implement TransformerLanguageModel first")
def test_transformer_lm_shapes() -> None:
    model = TransformerLanguageModel(vocab_size=10, block_size=8, d_model=16, num_heads=2)
    input_ids = torch.randint(0, 10, (2, 8))
    logits, loss = model(input_ids, labels=input_ids)
    assert logits.shape == (2, 8, 10)
    assert loss is not None
