"""Launch the required char-level GRU experiment."""

from src.experiments import run_experiment

from pathlib import Path

ROOT_DIR = Path(__file__).resolve().parent.parent

if __name__ == "__main__":
    run_experiment(ROOT_DIR / "configs" / "char_gru.yaml")