import torch
from pathlib import Path
from src.config import load_config
from src.tokenizers import CharTokenizer
from src.lm import GRULanguageModel, TransformerLanguageModel
from src.generate import generate_text

# Определяем корневую директорию проекта
ROOT_DIR = Path(__file__).resolve().parent.parent


def main() -> None:
    # 1. Указываем пути к конфигу и лучшему чекпоинту
    config_path = ROOT_DIR / "configs" / "char_transformer.yaml"
    config = load_config(config_path)

    checkpoint_path = Path(config.training.checkpoint_dir) / f"{config.experiment_name}.pt"

    print(f"Загрузка конфигурации из {config_path.name}...")

    # 2. Настраиваем устройство (GPU/CPU)
    device = torch.device(config.training.device if torch.cuda.is_available() else "cpu")
    print(f"Используем устройство: {device}")

    # 3. Нам нужен токенизатор, чтобы переводить промпты в числа.
    # Так как мы используем CharTokenizer, нам нужно инициализировать его словарь.
    # Для символьного токенизатора проще всего «обучить» его заново на ходу,
    # чтобы он знал все символы (это занимает доли секунды).
    from datasets import load_dataset
    dataset = load_dataset(config.data.dataset_name)
    raw_texts = dataset["train"][config.data.text_column][:config.data.max_texts]

    tokenizer = CharTokenizer()
    tokenizer.train(raw_texts)
    print("Токенизатор успешно инициализирован.")

    # 4. Сборка архитектуры модели
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

    # 5. Загрузка сохраненных весов
    if not checkpoint_path.exists():
        raise FileNotFoundError(
            f"Чекпоинт не найден по пути: {checkpoint_path}. "
            f"Сначала запусти обучение через train_char_gru.py!"
        )

    print(f"Загрузка весов из {checkpoint_path}...")
    model.load_state_dict(torch.load(checkpoint_path, map_location=device, weights_only=True))
    model.to(device)
    model.eval()  # Обязательно переводим в режим оценки (выключает dropout!)

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
        generated = generate_text(model, tokenizer, prompt=prompt, max_tokens=125)
        print(f" Результат:\n{generated}")
        print("-" * 40)


if __name__ == "__main__":
    main()
