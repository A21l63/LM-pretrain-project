import torch
from pathlib import Path
from src.experiments import run_experiment

ROOT_DIR = Path(__file__).resolve().parent.parent

def main():
    config_path = ROOT_DIR / "configs" / "hf_tokenizer_gru.yaml"
    run_experiment(config_path)


if __name__ == "__main__":
    main()