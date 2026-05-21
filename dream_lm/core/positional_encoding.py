"""
Sinusoidal positional encoding.

Spec §2:
    P_t[pos, 2i]   = sin(pos / 10000^(2i/d_model))
    P_t[pos, 2i+1] = cos(pos / 10000^(2i/d_model))

Added to token embeddings before the first transformer layer.
No learned parameters — purely deterministic.
"""

import math

import torch
import torch.nn as nn
from torch import Tensor


class SinusoidalPositionalEncoding(nn.Module):
    """Sinusoidal positional encoding as in "Attention Is All You Need".

    Pre-computes a (max_seq_len, d_model) table of sin/cos encodings
    and adds them to input embeddings.
    """

    def __init__(self, d_model: int, max_seq_len: int = 512) -> None:
        super().__init__()
        self.d_model = d_model
        self.max_seq_len = max_seq_len

        # Build the encoding table
        pe = torch.zeros(max_seq_len, d_model)  # (max_seq_len, d_model)
        position = torch.arange(0, max_seq_len, dtype=torch.float).unsqueeze(1)  # (max_seq_len, 1)
        div_term = torch.exp(
            torch.arange(0, d_model, 2, dtype=torch.float) * (-math.log(10000.0) / d_model)
        )  # (d_model/2,)

        pe[:, 0::2] = torch.sin(position * div_term)
        # slice div_term for odd d_model: pe[:, 1::2] has floor(d_model/2) columns
        pe[:, 1::2] = torch.cos(position * div_term[:pe[:, 1::2].shape[1]])

        # Register as buffer so it moves with the model and isn't a parameter
        self.register_buffer("pe", pe.unsqueeze(0))  # (1, max_seq_len, d_model)

    def forward(self, x: Tensor, offset: int = 0) -> Tensor:
        """
        Args:
            x: (batch, seq_len, d_model)
            offset: starting position index (for incremental inference)

        Returns:
            x + positional_encoding: (batch, seq_len, d_model)
        """
        _, seq_len, _ = x.shape
        assert offset + seq_len <= self.max_seq_len, (
            f"Position offset {offset} + seq_len {seq_len} exceeds max_seq_len {self.max_seq_len}"
        )
        return x + self.pe[:, offset : offset + seq_len, :]
