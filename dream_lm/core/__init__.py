"""
DREAM-LM Core — Stage 1: Minimal Transformer

Causal self-attention, multi-head attention, positional encoding,
LayerNorm, residual connections, feed-forward block.

All implementations from scratch — no nn.MultiheadAttention.
"""

from dream_lm.core.attention import CausalAttention
from dream_lm.core.multihead import MultiHeadAttention
from dream_lm.core.positional_encoding import SinusoidalPositionalEncoding
from dream_lm.core.transformer_layer import TransformerLayer, FeedForward

__all__ = [
    "CausalAttention",
    "MultiHeadAttention",
    "SinusoidalPositionalEncoding",
    "TransformerLayer",
    "FeedForward",
]
