"""Configuration helpers for the tiny LM project."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any

import yaml


@dataclass
class DataConfig:
    dataset_name: str
    text_column: str
    max_texts: int
    validation_fraction: float


@dataclass
class TokenizerConfig:
    type: str
    model_name: str | None = None


@dataclass
class ModelConfig:
    type: str
    block_size: int
    d_model: int
    num_layers: int
    dropout: float
    gru_hidden_size: int | None = None
    num_heads: int | None = None


@dataclass
class TrainingConfig:
    batch_size: int
    max_steps: int
    learning_rate: float
    eval_interval: int
    grad_clip: float
    checkpoint_dir: str
    device: str = "cpu"


@dataclass
class GenerationConfig:
    prompt: str
    max_new_tokens: int
    temperature: float
    top_k: int


@dataclass
class ExperimentConfig:
    """Strict dataclass for the entire experiment configuration."""
    experiment_name: str
    seed: int
    data: DataConfig
    tokenizer: TokenizerConfig
    model: ModelConfig
    training: TrainingConfig
    generation: GenerationConfig


def load_config(path: str | Path) -> ExperimentConfig:
    """Load a YAML experiment configuration file and parse it into dataclasses."""
    path = Path(path)
    with path.open("r", encoding="utf-8") as f:
        raw_values = yaml.safe_load(f)

    train_raw = raw_values["training"]
    if "learning_rate" in train_raw:
        train_raw["learning_rate"] = train_raw.pop("learning_rate")
    if "checkpoint_path" in train_raw:
        path_val = train_raw.pop("checkpoint_path")
        train_raw["checkpoint_dir"] = str(Path(path_val).parent)

    model_raw = raw_values["model"]
    if "num_layers" in model_raw:
        model_raw["num_layers"] = model_raw.pop("num_layers")

    data_config = DataConfig(**raw_values["data"])
    tokenizer_config = TokenizerConfig(**raw_values["tokenizer"])
    model_config = ModelConfig(**model_raw)
    training_config = TrainingConfig(**train_raw)
    generation_config = GenerationConfig(**raw_values["generation"])

    return ExperimentConfig(
        experiment_name=raw_values["experiment_name"],
        seed=raw_values["seed"],
        data=data_config,
        tokenizer=tokenizer_config,
        model=model_config,
        training=training_config,
        generation=generation_config
    )