"""Launch the required char-level Transformer experiment."""

from src.experiments import run_experiment


if __name__ == "__main__":
    run_experiment("configs/char_transformer.yaml")
