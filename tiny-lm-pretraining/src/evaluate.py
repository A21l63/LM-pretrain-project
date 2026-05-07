"""Evaluation helpers for trained tiny language models."""

from __future__ import annotations


def compute_perplexity(loss: float) -> float:
    """Convert cross-entropy loss to perplexity.

    TODO(student): remember that perplexity comparisons across different
    tokenizers require careful interpretation.
    """
    ...


def summarize_generation_errors(samples: list[str]) -> list[str]:
    """Create notes about common generation errors for the final report."""
    ...
