"""
Predictive coding utilities for DREAM-LM.

Spec §6.1:
    Perception error measures the gap between what the model's hidden state
    encodes and what it actually predicted — a self-prediction error that
    works at both training and inference time.

    eps_perc = h_final - W_vocab^T @ y_hat

where y_hat = softmax(logits) and W_vocab^T projects predictions back
into embedding space via einsum("vd,bsv->bsd", W, y_hat).
"""

import torch
from torch import Tensor


def compute_perception_error(
    logits: Tensor,
    h_final: Tensor,
    w_vocab_weight: Tensor,
) -> Tensor:
    """Compute perception error epsilon^perc.

    Spec §6.1:
        y_hat = softmax(logits)
        eps_perc = h_final - W_vocab^T @ y_hat

    The einsum handles the batch/seq broadcast correctly:
        W_vocab: (vocab_size, d_model)
        y_hat:   (batch, seq_len, vocab_size)
        result:  (batch, seq_len, d_model)

    Args:
        logits: (batch, seq_len, vocab_size) — raw output from W_vocab
        h_final: (batch, seq_len, d_model) — hidden state after final LayerNorm
        w_vocab_weight: (vocab_size, d_model) — W_vocab weight matrix

    Returns:
        eps_perc: (batch, seq_len, d_model) — perception error tensor.
            Gradients flow through to h_final and logits (via y_hat).
            Caller decides whether to detach.
    """
    y_hat = torch.softmax(logits, dim=-1)  # (batch, seq_len, vocab_size)

    # Project predictions back into embedding space:
    # W^T @ y_hat where W: (vocab_size, d_model), y_hat: (batch, seq_len, vocab_size)
    # einsum: v=vocab_size, d=d_model, b=batch, s=seq_len
    # "vd,bsv->bsd" contracts vocab dimension, outputs (batch, seq_len, d_model)
    w_vocab_T_y = torch.einsum("vd,bsv->bsd", w_vocab_weight, y_hat)

    eps_perc = h_final - w_vocab_T_y  # (batch, seq_len, d_model)
    return eps_perc


def perception_error_norm(eps_perc: Tensor) -> Tensor:
    """Compute L2 norm of perception error per position.

    Args:
        eps_perc: (batch, seq_len, d_model)

    Returns:
        norm: (batch, seq_len) — per-position L2 norm
    """
    return torch.linalg.vector_norm(eps_perc, dim=-1)  # (batch, seq_len)
