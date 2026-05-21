"""
Feed-forward block and complete transformer layer.

Feed-forward: two linear layers with configurable activation, internal
expansion factor (default 4×). Activation is a string from ACTIVATIONS
so it maps directly to YAML config values.

Transformer layer: LayerNorm → MultiHeadAttention → residual →
LayerNorm → FFN → residual.

Pre-norm architecture (LayerNorm before attention and FFN).
"""

import torch.nn as nn
from torch import Tensor

from dream_lm.core.multihead import MultiHeadAttention

# Supported activations — extend here when adding SwiGLU in later stages
ACTIVATIONS: dict[str, type[nn.Module]] = {
    "relu": nn.ReLU,
    "gelu": nn.GELU,
}


class FeedForward(nn.Module):
    """Position-wise feed-forward network.

    FFN(x) = W_2 · act(W_1 · x + b_1) + b_2

    Internal dimension is ff_mult × d_model (default 4×).
    Activation is controlled by the 'activation' string so it maps
    directly to config values without code changes.
    """

    def __init__(self, d_model: int, ff_mult: int = 4, activation: str = "relu") -> None:
        super().__init__()
        if activation not in ACTIVATIONS:
            raise ValueError(f"Unknown activation '{activation}'. Choose from {list(ACTIVATIONS)}")
        d_ff = d_model * ff_mult
        self.w_1 = nn.Linear(d_model, d_ff)
        self.w_2 = nn.Linear(d_ff, d_model)
        self.act = ACTIVATIONS[activation]()

        nn.init.xavier_uniform_(self.w_1.weight)
        nn.init.xavier_uniform_(self.w_2.weight)

    def forward(self, x: Tensor) -> Tensor:
        """
        Args:
            x: (batch, seq_len, d_model)

        Returns:
            output: (batch, seq_len, d_model)
        """
        return self.w_2(self.act(self.w_1(x)))


class TransformerLayer(nn.Module):
    """Single pre-norm transformer layer.

    Architecture:
        x → LayerNorm → MultiHeadAttention → +x → LayerNorm → FFN → +x

    Pre-norm is more stable than post-norm for deep stacks.
    """

    def __init__(
        self,
        d_model: int,
        n_heads: int,
        ff_mult: int = 4,
        max_seq_len: int = 512,
        activation: str = "relu",
    ) -> None:
        super().__init__()
        self.d_model = d_model

        self.ln_1 = nn.LayerNorm(d_model)
        self.attn = MultiHeadAttention(d_model, n_heads, max_seq_len)
        self.ln_2 = nn.LayerNorm(d_model)
        self.ffn = FeedForward(d_model, ff_mult, activation)

    def forward(self, x: Tensor) -> tuple[Tensor, Tensor]:
        """
        Args:
            x: (batch, seq_len, d_model)

        Returns:
            output: (batch, seq_len, d_model)
            attn_weights: (batch, n_heads, seq_len, seq_len)
        """
        h = self.ln_1(x)
        attn_out, attn_weights = self.attn(h)
        x = x + attn_out

        h = self.ln_2(x)
        ffn_out = self.ffn(h)
        x = x + ffn_out

        return x, attn_weights
