from __future__ import annotations

import torch


def top_k_filter(logits: torch.Tensor, top_k: int | None) -> torch.Tensor:
    if top_k is None or top_k <= 0:
        return logits

    top_k_values, _ = torch.topk(logits, top_k, dim=-1)
    threshold = top_k_values[..., -1, None]
    mask = logits < threshold
    logits = logits.masked_fill(mask, float('-inf'))
    return logits


def generate_text(
        model: torch.nn.Module,
        tokenizer,
        prompt: str,
        max_tokens: int = 100,
        temperature: float = 1.0,
        top_k: int | None = None,
        device: torch.device | None = None,
) -> str:
    if device is None:
        device = next(model.parameters()).device

    input_ids = tokenizer.encode(prompt)
    generated = torch.tensor(input_ids, dtype=torch.long, device=device).unsqueeze(0)

    eos_token_id = getattr(tokenizer, 'eos_token_id', None)

    model.eval()
    with torch.no_grad():
        for _ in range(max_tokens):
            context = generated[:, :]

            logits, _ = model(context)
            next_logits = logits[0, -1, :]

            if temperature > 0:
                next_logits = next_logits / temperature
            else:
                next_token_id = torch.argmax(next_logits, dim=-1, keepdim=True)
                generated = torch.cat([generated, next_token_id.unsqueeze(0)], dim=1)
                if eos_token_id is not None and next_token_id.item() == eos_token_id:
                    break
                continue

            if top_k is not None:
                next_logits = top_k_filter(next_logits, top_k)

            probs = torch.softmax(next_logits, dim=-1)
            next_token_id = torch.multinomial(probs, num_samples=1)

            generated = torch.cat([generated, next_token_id.unsqueeze(0)], dim=1)

            if eos_token_id is not None and next_token_id.item() == eos_token_id:
                break

    generated_ids = generated[0].tolist()
    output_text = tokenizer.decode(generated_ids)
    return output_text
