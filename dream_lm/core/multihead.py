"""
Multi-head causal attention.

Each head: projects Q,K,V to d_head, runs causal attention, outputs d_head.
Concatenated output: n_heads × d_head = d_model.

Spec §3 — multi-head variant of scaled dot-product attention.
"""

import math

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch import Tensor


class MultiHeadAttention(nn.Module):
    """Multi-head causal attention implemented from scratch.

    Q, K, V are projected from d_model → d_model, reshaped into
    (batch, n_heads, seq_len, d_head), then attention is computed
    in parallel across heads.
    """

    def __init__(self, d_model: int, n_heads: int, max_seq_len: int = 512) -> None:
        super().__init__()
        assert d_model % n_heads == 0, f"d_model={d_model} not divisible by n_heads={n_heads}"
        self.d_model = d_model
        self.n_heads = n_heads
        self.d_head = d_model // n_heads

        self.w_q = nn.Linear(d_model, d_model, bias=False)
        self.w_k = nn.Linear(d_model, d_model, bias=False)
        self.w_v = nn.Linear(d_model, d_model, bias=False)
        self.w_o = nn.Linear(d_model, d_model, bias=False)

        for proj in (self.w_q, self.w_k, self.w_v, self.w_o):
            nn.init.xavier_uniform_(proj.weight)

        # Pre-built causal mask as bool buffer — avoids allocation on every forward
        mask = torch.tril(torch.ones(max_seq_len, max_seq_len, dtype=torch.bool))
        self.register_buffer("causal_mask", mask.unsqueeze(0).unsqueeze(0))  # (1, 1, max_seq_len, max_seq_len)

    def forward(self, x: Tensor) -> tuple[Tensor, Tensor]:
        """
        Args:
            x: (batch, seq_len, d_model)

        Returns:
            output: (batch, seq_len, d_model)
            attn_weights: (batch, n_heads, seq_len, seq_len)
        """
        batch, seq_len, _ = x.shape

        # Project and reshape for multi-head
        q = self.w_q(x).view(batch, seq_len, self.n_heads, self.d_head).transpose(1, 2)
        k = self.w_k(x).view(batch, seq_len, self.n_heads, self.d_head).transpose(1, 2)
        v = self.w_v(x).view(batch, seq_len, self.n_heads, self.d_head).transpose(1, 2)
        # q, k, v: (batch, n_heads, seq_len, d_head)

        # Scaled dot-product
        scores = torch.matmul(q, k.transpose(-2, -1)) / math.sqrt(self.d_head)
        # scores: (batch, n_heads, seq_len, seq_len)

        mask = self.causal_mask[:, :, :seq_len, :seq_len]  # (1, 1, seq_len, seq_len)
        scores = scores.masked_fill(~mask, float("-inf"))

        # Attend
        attn_weights = F.softmax(scores, dim=-1)
        output = torch.matmul(attn_weights, v)
        # output: (batch, n_heads, seq_len, d_head)

        # Concatenate heads and project
        output = output.transpose(1, 2).contiguous().view(batch, seq_len, self.d_model)
        output = self.w_o(output)

        return output, attn_weights
