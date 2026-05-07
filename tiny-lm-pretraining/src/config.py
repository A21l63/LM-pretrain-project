"""Configuration helpers for the tiny LM project.

Students should keep configuration separate from model code so the same
training script can launch GRU and Transformer experiments.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any

import yaml


@dataclass
class ExperimentConfig:
    """Thin wrapper around a nested config dictionary.

    TODO(student): optionally replace this with stricter dataclasses for every
    config section after you understand which fields are required.
    """

    values: dict[str, Any]

    def get(self, key: str, default: Any = None) -> Any:
        ...


def load_config(path: str | Path) -> ExperimentConfig:
    """Load a YAML experiment configuration file."""
    path = Path(path)
    with path.open("r", encoding="utf-8") as f:
        values = yaml.safe_load(f)
    return ExperimentConfig(values=values)
