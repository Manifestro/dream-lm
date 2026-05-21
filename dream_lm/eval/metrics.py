"""
Evaluation metrics for language modeling.

Perplexity is the standard metric — exp(cross_entropy) measures
the effective branching factor of the model's predictions.

Spec §5:
    L_LM = -sum log p(x_{t+1} | x_{1:t})
    perplexity = exp(L_LM / N)
"""

import math

import torch
from torch import Tensor


def perplexity(logits: Tensor, targets: Tensor) -> float:
    """Compute perplexity from logits and target token IDs.

    Args:
        logits: (batch, seq_len, vocab_size) — model predictions
        targets: (batch, seq_len) — ground truth token IDs

    Returns:
        perplexity value (lower is better, random baseline = vocab_size)
    """
    loss = torch.nn.functional.cross_entropy(
        logits.view(-1, logits.size(-1)),
        targets.view(-1),
    )
    return torch.exp(loss).item()


def bits_per_char(logits: Tensor, targets: Tensor) -> float:
    """Compute bits per character — alternative metric for char-level LM.

    bpc = cross_entropy / ln(2)
    For random char-level model with vocab_size=65: bpc ≈ log2(65) ≈ 6.02

    Args:
        logits: (batch, seq_len, vocab_size)
        targets: (batch, seq_len)

    Returns:
        bits per character
    """
    loss = torch.nn.functional.cross_entropy(
        logits.view(-1, logits.size(-1)),
        targets.view(-1),
    )
    return (loss / math.log(2)).item()
