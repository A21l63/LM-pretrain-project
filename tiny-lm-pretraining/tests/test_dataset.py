"""Tests students should make pass while implementing LMDataset."""

import pytest

torch = pytest.importorskip("torch")

from src.data import LMDataset


def test_lm_dataset_shift() -> None:
    dataset = LMDataset([0, 1, 2, 3, 4], block_size=4)
    x, y = dataset[0]
    assert torch.equal(x, torch.tensor([0, 1, 2, 3]))
    assert torch.equal(y, torch.tensor([1, 2, 3, 4]))


def test_lm_dataset_length() -> None:
    dataset = LMDataset([0, 1, 2, 3, 4, 5], block_size=3)
    assert len(dataset) == 3
