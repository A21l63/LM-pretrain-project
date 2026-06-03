import torch
from pathlib import Path
from src.config import load_config
from src.tokenizers import CharTokenizer
from src.tokenizers import HFTokenizerWrapper
from src.lm import GRULanguageModel, TransformerLanguageModel
from src.generate import generate_text

ROOT_DIR = Path(__file__).resolve().parent.parent


def main() -> None:
    config_path = ROOT_DIR / "configs" / "hf_tokenizer_transformer.yaml"
    config = load_config(config_path)

    checkpoint_path = Path(config.training.checkpoint_dir) / f"{config.experiment_name}.pt"

    print(f"Загрузка конфигурации из {config_path.name}...")

    device = torch.device(config.training.device if torch.cuda.is_available() else "cpu")
    print(f"Используем устройство: {device}")

    from datasets import load_dataset
    dataset = load_dataset(config.data.dataset_name)
    raw_texts = dataset["train"][config.data.text_column][:config.data.max_texts]

    tokenizer = HFTokenizerWrapper(name=config.tokenizer.model_name)
    vocab_size = tokenizer.vocab_size
    print("Токенизатор успешно инициализирован.")

    print("Сборка модели...")
    if config.model.type == "transformer":
        model = TransformerLanguageModel(
            vocab_size=tokenizer.vocab_size,
            block_size=config.model.block_size,
            d_model=config.model.d_model,
            num_layers=config.model.num_layers,
            num_heads=config.model.num_heads,
        )
    elif config.model.type == "gru":
        model = GRULanguageModel(
            vocab_size=tokenizer.vocab_size,
            block_size=config.model.block_size,
            d_model=config.model.d_model,
            num_layers=config.model.num_layers,
        )
    else:
        raise ValueError(f"Unknown model type: {config.model.type}")

    if not checkpoint_path.exists():
        raise FileNotFoundError(
            f"Чекпоинт не найден по пути: {checkpoint_path}. "
            f"Сначала запусти обучение через train_char_gru.py!"
        )

    print(f"Загрузка весов из {checkpoint_path}...")
    model.load_state_dict(torch.load(checkpoint_path, map_location=device, weights_only=True))
    model.to(device)
    model.eval()

    prompts = [
        "Once upon a time",
        "A little girl named Lucy",
        "The sun was shining",
        "Suddenly, a big dog"
    ]

    print("\n" + "="*40)
    print("🤖 ЗАПУСК ГЕНЕРАЦИИ ТЕКСТА 🤖")
    print("="*40)

    for prompt in prompts:
        print(f"\n Промпт: \"{prompt}\"")
        generated = generate_text(model, tokenizer, prompt=prompt, max_tokens=230)
        print(f" Результат:\n{generated}")
        print("-" * 40)


if __name__ == "__main__":
    main()
