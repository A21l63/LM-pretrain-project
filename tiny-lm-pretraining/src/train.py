"""Training and validation loops for tiny language models."""

from __future__ import annotations

from pathlib import Path
from typing import Any

import torch
from torch.utils.data import DataLoader
from tqdm import tqdm


def get_device() -> torch.device:
    """Choose CUDA if available, otherwise CPU."""
    if torch.cuda.is_available():
        return torch.device("cuda")
    return torch.device("cpu")


def train_one_step(
    model: torch.nn.Module,
    batch: tuple[torch.Tensor, torch.Tensor], #два тензора так как мы передаём текст и правильные ответы
    optimizer: torch.optim.Optimizer,
    device: torch.device,
    grad_clip: float | None = None,
) -> float:
    """Run one optimization step and return the loss value."""
    model.train()
    input_ids, labels = batch
    input_ids = input_ids.to(device)
    labels = labels.to(device)
    optimizer.zero_grad()
    _, loss = model(input_ids, labels)
    loss.backward()
    if grad_clip is not None: #нормирует градиенты если они слишком большие
        torch.nn.utils.clip_grad_norm_(model.parameters(), grad_clip)
    optimizer.step() #сам оптимайзер обновляет веса, самый лёгкий способ: w = w - n*g, где w - вес, n - шаг обучения, g - значение градиента которое посчитали в backward
    return loss.item() #просто преобразовывает тензор в float, что бы не было переполнения памяти

@torch.no_grad()
def evaluate_loss( #режим экзамена, даём новые данные, которые нейросетка никогда не видела
    model: torch.nn.Module,
    dataloader: DataLoader, #новый датасет
    device: torch.device,
    max_batches: int | None = None,
) -> float:
    """Compute average validation loss."""
    model.eval()
    total_loss = 0.0
    batches_counted = 0
    for batch in dataloader:
        if max_batches is not None and batches_counted >= max_batches:
            break
        input_ids, labels = batch
        input_ids = input_ids.to(device)
        labels = labels.to(device)
        _, loss = model(input_ids, labels)
        total_loss += loss.item()
        batches_counted += 1
    return total_loss / batches_counted



def train_model(
    model: torch.nn.Module,
    train_loader: DataLoader,
    val_loader: DataLoader | None,
    optimizer: torch.optim.Optimizer,
    max_steps: int,
    device: torch.device | None = None,
    eval_interval: int = 100,
    grad_clip: float | None = None,
    checkpoint_path: str | Path | None = None,
) -> list[dict[str, Any]]:
    """Train any model that implements model(input_ids, labels)."""
    if device is None:
        device = get_device()
    model.to(device)
    history: list[dict[str, Any]] = [] #список из словарей, словарь в формате название метрики - значение
    step = 0
    best_val_loss = float("inf")
    def batch_generator():
        while True:
            for batch in train_loader:
                yield batch
    train_iter = batch_generator()
    pbar = tqdm(total=max_steps, desc="Training")
    while step < max_steps:
        batch = next(train_iter)
        train_loss = train_one_step(
            model=model,
            batch=batch,
            optimizer=optimizer,
            device=device,
            grad_clip=grad_clip
        )
        step += 1
        pbar.update(1)
        if step % eval_interval == 0 or step == max_steps:
            val_loss = None

            if val_loader is not None:
                val_loss = evaluate_loss(
                    model=model,
                    dataloader=val_loader,
                    device=device,
                    max_batches=20
                )

            log_entry = {
                "step": step,
                "train_loss": train_loss,
                "val_loss": val_loss,
            }
            history.append(log_entry)
            status_text = f"Step {step} | Train Loss: {train_loss:.4f}"
            if val_loss is not None:
                status_text += f" | Val Loss: {val_loss:.4f}"
            pbar.set_postfix_str(status_text)

            if checkpoint_path is not None and val_loss is not None:
                if val_loss < best_val_loss:
                    best_val_loss = val_loss
                    torch.save(model.state_dict(), checkpoint_path)
    pbar.close()
    return history