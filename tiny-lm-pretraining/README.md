# Tiny LM Pretraining: маленькие языковые модели с нуля

## 1. Краткое описание проекта

В этом проекте вы реализуете минимальный pipeline предобучения маленькой языковой модели на корпусе коротких историй, например на подмножестве TinyStories. Цель — не получить «настоящий ChatGPT», а понять, из каких частей состоит language model pretraining и как данные проходят через модель.

Модель обучается задаче предсказания следующего токена:

```text
input:  t0, t1, t2, ..., t_{n-1}
target: t1, t2, t3, ..., t_n
```

Главная схема проекта:

```text
text → tokenizer → token ids → dataset → embeddings → backbone → lm_head → logits → loss → generation
```

Фокус проекта:

```text
text → tokens → token ids → embeddings → backbone → lm_head → logits → loss
```

Один и тот же высокоуровневый интерфейс модели должен использоваться и для GRU, и для Transformer. Эти варианты должны отличаться в основном модулем `backbone`.

Основное сравнение:

```text
char-level tokenizer + GRU backbone
char-level tokenizer + Transformer backbone
```

Дополнительное сравнение, если останется время:

```text
pretrained tokenizer + GRU backbone
pretrained tokenizer + Transformer backbone
```

Опциональное расширение:

```text
same or similar pipeline using Hugging Face Transformers
```

## 2. Учебные цели

К концу проекта вы должны уметь объяснить:

- как текст превращается в числа;
- что такое словарь токенизатора и почему его размер важен;
- как устроен dataset для next-token prediction;
- почему `x` и `y` сдвинуты на один токен;
- как выглядят tensor shapes на каждом этапе;
- что делают embeddings;
- что такое `backbone` и чем GRU отличается от decoder-only Transformer;
- почему языковая модель возвращает logits размера vocabulary;
- как считается cross-entropy loss;
- как работает autoregressive generation;
- почему validation loss для разных токенизаторов нужно сравнивать осторожно.

## 3. Pipeline языковой модели

Проект строится вокруг такой цепочки:

```text
text → tokenizer → token ids → dataset → embeddings → backbone → lm_head → logits → loss → generation
```

- `text`: исходные строки из корпуса коротких историй.
- `tokenizer`: правило преобразования текста в токены.
- `token ids`: целые числа, соответствующие токенам.
- `dataset`: нарезает длинный поток token ids на пары `x` и `y`.
- `embeddings`: превращают ids в dense vectors.
- `backbone`: GRU или decoder-only Transformer обрабатывает последовательность.
- `lm_head`: линейный слой, который превращает hidden states в logits по словарю.
- `logits`: ненормированные оценки вероятностей следующего токена.
- `loss`: ошибка next-token prediction.
- `generation`: модель многократно предсказывает следующий токен и добавляет его к контексту.

## 4. Next-token dataset

Для последовательности token ids:

```text
[t0, t1, t2, t3, t4]
```

dataset должен вернуть:

```text
x = [t0, t1, t2, t3]
y = [t1, t2, t3, t4]
```

Идея: на позиции `i` модель видит токены до текущего места и учится предсказывать следующий токен.

## 5. Shape tracing

Ожидаемые формы тензоров:

```text
input_ids: [B, T]
embeddings: [B, T, D]
backbone output: [B, T, D]
logits: [B, T, V]
labels: [B, T]
loss input logits: [B*T, V]
loss input labels: [B*T]
```

Где:

- `B` — batch size;
- `T` — sequence length или `block_size`;
- `D` — `d_model`, размер embedding/hidden vectors;
- `V` — `vocab_size`, размер словаря.

Рекомендуется при отладке печатать shapes после каждого важного шага. Если shape не совпадает с ожиданием, сначала исправляйте shape, а уже потом loss или generation.

## 6. Организация команды

Команда может разделиться на зоны ответственности или работать над всеми частями вместе. Оба варианта допустимы, если к концу проекта все участники понимают полный pipeline.

### Вариант A: разделение по зонам ответственности

Можно разделиться на несколько направлений:

1. Tokenization and data preparation.
2. Dataset and batching.
3. Base language model interface and generation.
4. Training and evaluation loop.
5. Backbones: GRU and decoder-only Transformer.
6. Experiments, result tables, plots, and final presentation.

Ни одна зона не является «главнее» других: pipeline работает только тогда, когда все части корректно соединены.

### Вариант B: совместная работа над всем pipeline

Если команде удобнее работать коллективно, можно проходить этапы вместе:

1. Сначала реализовать character tokenizer и dataset.
2. Затем реализовать `BaseLanguageModel`.
3. Затем добавить GRU backbone.
4. Затем добавить Transformer backbone.
5. Затем запустить эксперименты.
6. Затем проанализировать результаты.

Этот вариант полностью допустим, если команда предпочитает совместную работу и каждый участник может объяснить все ключевые части.

## 7. Структура репозитория

```text
tiny-lm-pretraining/
  README.md
  requirements.txt
  pyproject.toml
  configs/
    char_gru.yaml
    char_transformer.yaml
    hf_tokenizer_gru.yaml
    hf_tokenizer_transformer.yaml
  src/
    __init__.py
    config.py
    tokenizers.py
    data.py
    lm.py
    backbones.py
    train.py
    evaluate.py
    generate.py
    experiments.py
    utils.py
  notebooks/
    01_data_and_tokenization.ipynb
    02_train_char_gru.ipynb
    03_train_char_transformer.ipynb
    04_compare_results.ipynb
  scripts/
    train_char_gru.py
    train_char_transformer.py
    generate_samples.py
    run_experiments.py
  results/
    .gitkeep
  tests/
    test_tokenizers.py
    test_dataset.py
    test_shapes.py
```

Репозиторий можно использовать двумя способами:

1. как script-based project через файлы в `scripts/`;
2. как notebook-based project через файлы в `notebooks/`.

## 8. Установка

Создайте виртуальное окружение и установите зависимости:

```bash
cd tiny-lm-pretraining
python -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
```

Проверьте, что Python видит пакет:

```bash
python -m pytest
```

В начале проекта многие тесты могут быть помечены как TODO/skip. По мере реализации вы можете снимать `skip` и заставлять тесты проходить.

## 9. Реализационные задачи

### 9.1 Tokenizers

В `src/tokenizers.py` задан общий интерфейс:

```python
class BaseTokenizer:
    def train(self, texts: list[str]) -> None:
        raise NotImplementedError

    def encode(self, text: str) -> list[int]:
        raise NotImplementedError

    def decode(self, ids: list[int]) -> str:
        raise NotImplementedError

    @property
    def vocab_size(self) -> int:
        raise NotImplementedError
```

Нужно реализовать `CharTokenizer`:

- собрать множество символов из обучающих текстов;
- построить отображения `stoi` и `itos`;
- реализовать `encode`;
- реализовать `decode`;
- проверить `decode(encode(text))`.

Опционально можно реализовать `HFTokenizerWrapper`, например вокруг GPT-2 tokenizer из Hugging Face Transformers. Важно: pretrained tokenizer меняет размер словаря и среднюю длину последовательности. Большой словарь делает `lm_head` намного больше, потому что модель должна классифицировать каждый hidden state в один из `V` токенов.

### 9.2 Dataset and batching

В `src/data.py` нужно реализовать `LMDataset`, который возвращает пары `x` и `y`, сдвинутые на один токен. Затем нужно создать `DataLoader`, который собирает много таких пар в batch.

Проверяйте:

```text
x.shape == [B, T]
y.shape == [B, T]
```

### 9.3 Base language model interface

В `src/lm.py` нужно реализовать общий класс `BaseLanguageModel`:

- token embedding layer;
- backbone;
- language modeling head;
- `forward(input_ids, labels=None)`;
- loss computation;
- `generate(input_ids, max_new_tokens, temperature=1.0, top_k=None)`;
- опционально `generate_text(prompt, tokenizer, max_new_tokens=100, ...)`;
- parameter counting utility.

Модель должна работать с token ids. Tokenization концептуально отделена от neural network, даже если tokenizer хранится внутри объекта для удобства.

### 9.4 GRU backbone

В `src/backbones.py` нужно реализовать `GRUBackbone` на основе `torch.nn.GRU`. Он должен принимать `[B, T, D]` и возвращать `[B, T, D]`. Если `hidden_size != d_model`, добавьте projection обратно в `d_model`.

### 9.5 Decoder-only Transformer backbone

Нужно реализовать `TransformerBlock` и `TransformerBackbone`:

- LayerNorm;
- causal self-attention;
- residual connection;
- feed-forward MLP;
- residual connection;
- dropout.

Используйте только decoder-only architecture. Encoder-decoder Transformer здесь не нужен, потому что задача — causal language modeling.

### 9.6 Training and validation

В `src/train.py` нужно реализовать:

- training loop;
- validation loop;
- loss logging;
- device handling;
- optional gradient clipping;
- optional checkpoint saving.

Обязательное поведение:

```text
model(input_ids, labels) returns logits and loss
```

Training должен печатать или сохранять:

- train loss;
- validation loss;
- current step;
- опционально generated samples.

### 9.7 Generation

В `src/generate.py` и/или методе `model.generate` нужно реализовать генерацию. Generation работает так: модель получает prompt, предсказывает следующий токен, добавляет его к контексту, затем повторяет этот процесс много раз.

Поддержите:

- sampling;
- temperature;
- top-k sampling, если успеете;
- greedy decoding, если хотите.

Важно: во время generation используйте только последние `block_size` токенов контекста.

## 10. Как запускать обучение

После реализации TODO можно запускать required experiments:

```bash
cd tiny-lm-pretraining
python scripts/train_char_gru.py
python scripts/train_char_transformer.py
```

Или напрямую через общий runner:

```bash
python scripts/run_experiments.py
```

Конфиги находятся в `configs/`. Начинайте с маленьких значений:

```text
block_size: 128
batch_size: 32
d_model: 128
max_steps: 1000
```

Если всё работает медленно на CPU, уменьшите `batch_size`, `d_model`, `num_layers` или `max_steps`.

## 11. Как генерировать текст

После обучения и сохранения checkpoint можно использовать:

```bash
python scripts/generate_samples.py
```

Рекомендуется генерировать тексты из одинаковых prompts для всех моделей, например:

```text
Once upon a time
The little cat
In a small village
```

Так сравнение будет честнее.

## 12. Обязательные эксперименты

Нужно провести минимум два эксперимента:

```text
1. char-level tokenizer + GRU backbone
2. char-level tokenizer + Transformer backbone
```

Для каждого эксперимента сохраните:

- конфиг;
- train loss curve;
- validation loss;
- число параметров;
- время обучения;
- generated samples.

## 13. Дополнительные эксперименты

Если останется время:

```text
3. pretrained tokenizer + GRU backbone
4. pretrained tokenizer + Transformer backbone
```

Сравните:

- vocabulary size;
- average sequence length in tokens;
- number of model parameters;
- training loss curve;
- validation loss;
- training time;
- generated samples for the same prompts;
- typical generation errors.

Validation loss across different tokenizers should be interpreted carefully, because different tokenizers define “one token” differently. Например, один char-token и один GPT-style token несут разное количество информации.

## 14. Опциональное расширение Hugging Face Transformers

Можно добавить high-level comparison через Hugging Face Transformers, но это не замена custom PyTorch pipeline. Не используйте pretrained GPT model как основной результат. Если делаете расширение, создайте маленькую модель с нуля из config, например:

```python
GPT2Config(
    vocab_size=tokenizer.vocab_size,
    n_positions=128,
    n_embd=128,
    n_layer=2,
    n_head=2,
)
```

Это нужно только для сравнения уровня абстракции: насколько больше работы вы делаете вручную в собственной реализации.

## 15. План на 2.5 недели

### Days 1–2: Minimal vertical slice

Цель:

```text
text → char tokenizer → token ids → dataset → batch → loss
```

Ожидаемый результат:

- `CharTokenizer` works;
- `LMDataset` works;
- `x` and `y` shapes are correct;
- a small model can compute finite loss.

### Days 3–4: BaseLanguageModel and GRU

Цель:

```text
char tokenizer → GRU LM → training loop → generation
```

Ожидаемый результат:

- loss decreases;
- validation loss is computed;
- generation works before and after training.

### Days 5–7: Decoder-only Transformer

Цель:

```text
same pipeline → Transformer backbone
```

Ожидаемый результат:

- causal mask works;
- Transformer output shapes are correct;
- Transformer trains with the same training loop.

### Days 8–9: Tokenization extension

Цель:

```text
pretrained tokenizer → same dataset interface → same models
```

Ожидаемый результат:

- compare char-level and pretrained tokenization;
- report vocabulary size and average sequence length.

### Days 10–12: Experiments

Цель:

- run final experiments;
- save loss curves;
- generate samples from the same prompts;
- compare GRU and Transformer.

### Final days: Analysis and presentation

Цель:

- prepare result tables;
- analyze typical errors;
- write conclusions;
- prepare short presentation.

## 16. Ожидаемые финальные результаты

Финальный проект должен включать:

1. Working code.
2. Short explanation of the pipeline.
3. Training logs or saved loss curves.
4. At least two trained models:
   - char + GRU;
   - char + Transformer.
5. Generated text samples from identical prompts.
6. Comparison table.
7. Error analysis.
8. Short final presentation.

## 17. Критерии оценивания

Возможные критерии:

- корректность tokenizer и dataset;
- правильный input-target shift;
- единый интерфейс для GRU и Transformer;
- корректные tensor shapes;
- корректная causal mask в Transformer;
- работающий training loop;
- validation loss считается отдельно от train loss;
- generation действительно autoregressive;
- эксперименты воспроизводимы через configs и seed;
- сравнение моделей содержит не только числа, но и анализ ошибок;
- команда может объяснить код, который сдала.

## 18. Debugging checklist

Проверяйте по списку:

- Does `decode(encode(text))` approximately reconstruct the text?
- Are `x` and `y` shifted by one token?
- Are tensor shapes correct?
- Is `loss` finite?
- Does train loss decrease?
- Is the model on the correct device?
- Are labels in the range `[0, vocab_size)`?
- Is the causal mask preventing attention to future tokens?
- Does generation use only the last `block_size` tokens?
- Are different models trained on the same train/validation split?

Если что-то сломалось, начните с маленького искусственного примера: одна строка текста, маленький vocab, batch size 1 или 2, block size 4 или 8.

## 19. Ответственное использование LLMs

Вы можете использовать LLMs для помощи с кодом, отладкой и объяснениями. Но команда обязана понимать и уметь объяснить:

- tensor shapes;
- input-target shift;
- loss computation;
- tokenizer behavior;
- model forward pass;
- generation loop;
- differences between GRU and Transformer backbones.

Любой скопированный код должен быть explainable by the team. Если вы не можете объяснить строку кода, значит её нельзя просто оставлять как «магическую» часть решения. Попросите LLM объяснить её проще, перепишите своими словами и проверьте на маленьком примере.

## 20. Важное напоминание

Не ожидайте, что маленькая модель за короткое время на CPU будет генерировать идеальные истории. Главная цель проекта — понять pretraining pipeline, научиться проверять shapes, запускать controlled experiments и честно анализировать результаты.
