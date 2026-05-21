"""
Causal self-attention with scaled dot-product.

Spec §3:
    A_t = softmax(Q_t K_{1:t}^T / sqrt(d_k) + M_causal)
    h_t = A_t · V_{1:t}

No nn.MultiheadAttention — implemented from scratch.
"""

import math

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch import Tensor


class CausalAttention(nn.Module):
    """Single-head scaled dot-product attention with causal masking.

    Computes attention over the sequence dimension, ensuring each token
    can only attend to itself and previous tokens.

    Spec §3 — single-head, standalone variant.
    """

    def __init__(self, d_model: int, max_seq_len: int = 512) -> None:
        super().__init__()
        self.d_model = d_model
        self.d_k = d_model

        self.w_q = nn.Linear(d_model, d_model, bias=False)
        self.w_k = nn.Linear(d_model, d_model, bias=False)
        self.w_v = nn.Linear(d_model, d_model, bias=False)

        for proj in (self.w_q, self.w_k, self.w_v):
            nn.init.xavier_uniform_(proj.weight)

        # Pre-built causal mask as bool buffer — avoids allocation on every forward
        mask = torch.tril(torch.ones(max_seq_len, max_seq_len, dtype=torch.bool))
        self.register_buffer("causal_mask", mask.unsqueeze(0))  # (1, max_seq_len, max_seq_len)

    def forward(self, x: Tensor) -> tuple[Tensor, Tensor]:
        """
        Args:
            x: (batch, seq_len, d_model)

        Returns:
            output: (batch, seq_len, d_model)
            attn_weights: (batch, seq_len, seq_len)
        """
        batch, seq_len, _ = x.shape

        q = self.w_q(x)  # (batch, seq_len, d_k)
        k = self.w_k(x)  # (batch, seq_len, d_k)
        v = self.w_v(x)  # (batch, seq_len, d_k)

        scores = torch.bmm(q, k.transpose(1, 2)) / math.sqrt(self.d_k)
        # scores: (batch, seq_len, seq_len)

        mask = self.causal_mask[:, :seq_len, :seq_len]  # (1, seq_len, seq_len)
        scores = scores.masked_fill(~mask, float("-inf"))

        attn_weights = F.softmax(scores, dim=-1)  # (batch, seq_len, seq_len)
        output = torch.bmm(attn_weights, v)        # (batch, seq_len, d_model)

        return output, attn_weights
