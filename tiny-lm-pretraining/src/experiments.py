"""Experiment runner skeleton for comparing tiny LM variants."""

from __future__ import annotations
import torch
from torch.utils.data import DataLoader
import gc
from src.tokenizers import CharTokenizer, HFTokenizerWrapper
from pathlib import Path
from src.data import LMDataset
from src.config import load_config
from src.lm import GRULanguageModel, TransformerLanguageModel
from src.train import train_model
from src.generate import generate_text
from datasets import load_dataset


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
    print("Загрузка датасета...")
    config = load_config(config_path)
    dataset = load_dataset(config.data.dataset_name)
    raw_texts = dataset["train"][config.data.text_column][:config.data.max_texts]
    if config.tokenizer.type == "char":
        tokenizer = CharTokenizer()
        tokenizer.train(raw_texts)
    elif config.tokenizer.type == "hf":
        tokenizer = HFTokenizerWrapper(config.tokenizer.model_name)
    else:
        raise ValueError(f"Unknown tokenizer type: {config.tokenizer.type}")

    val_size = int(len(raw_texts) * config.data.validation_fraction)
    train_texts = raw_texts[val_size:]
    val_texts = raw_texts[:val_size]

    print(f"Размер обучающей выборки: {len(train_texts)} текстов")
    print(f"Размер валидационной выборки: {len(val_texts)} текстов")

    print("Токенизация тренировочных текстов...")
    train_tokens = []
    for text in train_texts:
        train_tokens.extend(tokenizer.encode(text))

    print("Токенизация валидационных текстов...")
    val_tokens = []
    for text in val_texts:
        val_tokens.extend(tokenizer.encode(text))

    train_dataset = LMDataset(token_ids=train_tokens, block_size=config.model.block_size)
    val_dataset = LMDataset(token_ids=val_tokens, block_size=config.model.block_size)

    print(f"Токенов в обучении: {len(train_tokens)}")
    print(f"Токенов в валидации: {len(val_tokens)}")

    del train_texts, val_texts, train_tokens, val_tokens

    gc.collect()
    torch.cuda.empty_cache()

    print("Память очищена, запускаем модель...")

    train_loader = DataLoader(
        train_dataset,
        batch_size=config.training.batch_size,
        shuffle=True
    )
    val_loader = DataLoader(
        val_dataset,
        batch_size=config.training.batch_size,
        shuffle=False
    )

    if config.model.type == "gru":
        model = GRULanguageModel(
            vocab_size=tokenizer.vocab_size,
            block_size=config.model.block_size,
            d_model=config.model.d_model,
            num_layers=config.model.num_layers,
        )
    elif config.model.type == "transformer":
        model = TransformerLanguageModel(
            vocab_size=tokenizer.vocab_size,
            block_size=config.model.block_size,
            d_model=config.model.d_model,
            num_layers=config.model.num_layers,
            num_heads=config.model.num_heads,
        )
    else:
        raise ValueError(f"Unknown model type: {config.model.type}")

    optimizer = torch.optim.AdamW(model.parameters(), lr=config.training.learning_rate)

    checkpoint_path = Path(config.training.checkpoint_dir) / f"{config.experiment_name}.pt"
    checkpoint_path.parent.mkdir(parents=True, exist_ok=True)

    device = torch.device(config.training.device)

    if checkpoint_path.exists():
        print(f"\nНайдено прошлое сохранение для {config.experiment_name}!")
        print(f"Загружаю веса из {checkpoint_path}...")
        model.load_state_dict(torch.load(checkpoint_path, map_location=device, weights_only=True))
    else:
        print(f"\nПрошлых сохранений для {config.experiment_name} не найдено. Начинаем с нуля.")

    model.to(device)

    history = train_model(
        model=model,
        train_loader=train_loader,
        val_loader=val_loader,
        optimizer=optimizer,
        max_steps=config.training.max_steps,
        device=torch.device(config.training.device),
        eval_interval=config.training.eval_interval,
        grad_clip=config.training.grad_clip,
        checkpoint_path=checkpoint_path,
    )

    val_losses = [log["val_loss"] for log in history if log["val_loss"] is not None]
    best_loss = min(val_losses) if val_losses else history[-1]["train_loss"]

    model.load_state_dict(torch.load(checkpoint_path, weights_only=True))  # Загружаем лучшие веса
    sample_text = generate_text(model, tokenizer, prompt="Once", max_tokens=100)
    print(f"\nСгенерированный текст:\n{sample_text}")

    metrics = {
        "experiment_name": config.experiment_name,
        "best_val_loss": best_loss,
        "final_train_loss": history[-1]["train_loss"],
        "generated_sample": sample_text,
    }
    try:
        import matplotlib.pyplot as plt

        steps = [log["step"] for log in history]
        train_losses = [log["train_loss"] for log in history]
        val_steps = [log["step"] for log in history if log["val_loss"] is not None]
        val_losses = [log["val_loss"] for log in history if log["val_loss"] is not None]

        plt.figure(figsize=(10, 5))
        plt.plot(steps, train_losses, label="Train Loss", color="#1f77b4")
        if val_losses:
            plt.plot(val_steps, val_losses, label="Val Loss", color="#ff7f0e", linestyle="--", marker="o")

        plt.title(f"История обучения: {config.experiment_name}")
        plt.xlabel("Шаги (Steps)")
        plt.ylabel("Loss")
        plt.grid(True, linestyle=":")
        plt.legend()

        plot_path = Path(config.training.checkpoint_dir) / f"{config.experiment_name}_loss.png"
        plt.savefig(plot_path, dpi=300)
        print(f"📈 График лосса успешно сохранен в {plot_path}")
        plt.close()
    except Exception as e:
        print(f"Не удалось построить график: {e}")

    return metrics


def run_many(config_paths: list[str | Path]) -> list[dict]:
    """Run several experiments and collect result dictionaries."""
    results = []

    for path in config_paths:
        print(f"\n{'=' * 10} Starting experiment: {path} {'=' * 10}")
        try:
            metrics = run_experiment(path)
            results.append(metrics)
            print(f"Finished! Best Val Loss: {metrics['best_val_loss']:.4f}")
        except Exception as e:
            print(f"Experiment {path} failed with error: {e}")

    return results
