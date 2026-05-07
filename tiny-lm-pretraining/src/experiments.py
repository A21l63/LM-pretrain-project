"""Experiment runner skeleton for comparing tiny LM variants."""

from __future__ import annotations

from pathlib import Path

from src.config import load_config


REQUIRED_EXPERIMENTS = [
    "configs/char_gru.yaml",
    "configs/char_transformer.yaml",
]

OPTIONAL_EXPERIMENTS = [
    "configs/hf_tokenizer_gru.yaml",
    "configs/hf_tokenizer_transformer.yaml",
]


def run_experiment(config_path: str | Path) -> dict:
    """Run one configured experiment and return metrics.

    TODO(student): load data, train tokenizer, create datasets/loaders, build
    the selected model, train it, generate samples, and save metrics.
    """
    config = load_config(config_path)
    ...


def run_many(config_paths: list[str | Path]) -> list[dict]:
    """Run several experiments and collect result dictionaries."""
    ...
