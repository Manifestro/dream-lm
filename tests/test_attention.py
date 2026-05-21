"""Tests for CausalAttention and MultiHeadAttention.

Three test types per block:
  1. test_output_shape — correct dimensions
  2. test_numerical_stability — no NaN/Inf on random input
  3. test_expected_behavior — causal mask works, future tokens not visible
"""

import torch

from dream_lm.core.attention import CausalAttention
from dream_lm.core.multihead import MultiHeadAttention


class TestCausalAttention:
    """Tests for single-head causal attention."""

    def test_output_shape(self) -> None:
        batch, seq_len, d_model = 4, 16, 64
        x = torch.randn(batch, seq_len, d_model)
        attn = CausalAttention(d_model)
        output, weights = attn(x)
        assert output.shape == (batch, seq_len, d_model)
        assert weights.shape == (batch, seq_len, seq_len)

    def test_numerical_stability(self) -> None:
        """No NaN/Inf on random input with various scales."""
        d_model = 128
        attn = CausalAttention(d_model)
        for scale in [0.01, 1.0, 100.0]:
            x = torch.randn(2, 32, d_model) * scale
            output, weights = attn(x)
            assert not torch.isnan(output).any(), f"NaN with scale={scale}"
            assert not torch.isinf(output).any(), f"Inf with scale={scale}"
            assert not torch.isnan(weights).any()
            assert not torch.isinf(weights).any()

    def test_causal_mask(self) -> None:
        """Future tokens must not be visible — attention weights above diagonal should be 0."""
        batch, seq_len, d_model = 1, 8, 64
        x = torch.randn(batch, seq_len, d_model)
        attn = CausalAttention(d_model)
        _, weights = attn(x)

        # Check upper triangle is zero
        upper_triangle = torch.triu(weights[0], diagonal=1)
        assert (upper_triangle.abs() < 1e-7).all(), "Causal mask failed — future tokens visible"

    def test_gradient_flows(self) -> None:
        """Gradients should flow through attention."""
        d_model = 32
        attn = CausalAttention(d_model)
        x = torch.randn(2, 8, d_model, requires_grad=True)
        output, _ = attn(x)
        loss = output.sum()
        loss.backward()
        assert x.grad is not None
        assert not torch.isnan(x.grad).any()
        # Check that parameters have gradients
        for name, param in attn.named_parameters():
            assert param.grad is not None, f"No gradient for {name}"


class TestMultiHeadAttention:
    """Tests for multi-head causal attention."""

    def test_output_shape(self) -> None:
        batch, seq_len, d_model, n_heads = 4, 16, 64, 4
        x = torch.randn(batch, seq_len, d_model)
        attn = MultiHeadAttention(d_model, n_heads)
        output, weights = attn(x)
        assert output.shape == (batch, seq_len, d_model)
        assert weights.shape == (batch, n_heads, seq_len, seq_len)

    def test_numerical_stability(self) -> None:
        """No NaN/Inf on random input."""
        d_model, n_heads = 128, 8
        attn = MultiHeadAttention(d_model, n_heads)
        for scale in [0.01, 1.0, 100.0]:
            x = torch.randn(2, 32, d_model) * scale
            output, weights = attn(x)
            assert not torch.isnan(output).any()
            assert not torch.isinf(output).any()
            assert not torch.isnan(weights).any()
            assert not torch.isinf(weights).any()

    def test_causal_mask(self) -> None:
        """All heads must respect causal mask."""
        batch, seq_len, d_model, n_heads = 1, 8, 64, 4
        x = torch.randn(batch, seq_len, d_model)
        attn = MultiHeadAttention(d_model, n_heads)
        _, weights = attn(x)

        # Check all heads: upper triangle should be zero
        for h in range(n_heads):
            upper = torch.triu(weights[0, h], diagonal=1)
            assert (upper.abs() < 1e-7).all(), f"Head {h} violates causal mask"

    def test_divisibility_assertion(self) -> None:
        """Should raise if d_model not divisible by n_heads."""
        import pytest
        with pytest.raises(AssertionError):
            MultiHeadAttention(64, 7)

    def test_gradient_flows(self) -> None:
        """Gradients should flow through all heads."""
        d_model, n_heads = 64, 4
        attn = MultiHeadAttention(d_model, n_heads)
        x = torch.randn(2, 8, d_model, requires_grad=True)
        output, _ = attn(x)
        loss = output.sum()
        loss.backward()
        assert x.grad is not None
        for name, param in attn.named_parameters():
            assert param.grad is not None, f"No gradient for {name}"
