"""Tests for positional encoding and transformer layer."""

import torch

from dream_lm.core.positional_encoding import SinusoidalPositionalEncoding
from dream_lm.core.transformer_layer import TransformerLayer, FeedForward


class TestSinusoidalPositionalEncoding:
    """Tests for sinusoidal positional encoding."""

    def test_output_shape(self) -> None:
        batch, seq_len, d_model = 4, 16, 64
        x = torch.randn(batch, seq_len, d_model)
        pe = SinusoidalPositionalEncoding(d_model)
        output = pe(x)
        assert output.shape == x.shape

    def test_numerical_stability(self) -> None:
        """PE should not introduce NaN/Inf."""
        d_model = 128
        pe = SinusoidalPositionalEncoding(d_model, max_seq_len=512)
        x = torch.randn(2, 64, d_model)
        output = pe(x)
        assert not torch.isnan(output).any()
        assert not torch.isinf(output).any()

    def test_deterministic(self) -> None:
        """PE should be the same for the same input — no learned params."""
        d_model = 64
        pe1 = SinusoidalPositionalEncoding(d_model)
        pe2 = SinusoidalPositionalEncoding(d_model)
        x = torch.randn(1, 10, d_model)
        assert torch.allclose(pe1.pe, pe2.pe)

    def test_no_learnable_parameters(self) -> None:
        """PE is deterministic — should have no trainable params."""
        pe = SinusoidalPositionalEncoding(64)
        trainable = sum(1 for p in pe.parameters() if p.requires_grad)
        assert trainable == 0, f"PE has {trainable} trainable params, expected 0"

    def test_max_seq_length_exceeded(self) -> None:
        """Should raise when seq_len > max_seq_len."""
        import pytest
        pe = SinusoidalPositionalEncoding(64, max_seq_len=10)
        x = torch.randn(1, 20, 64)
        with pytest.raises(AssertionError):
            pe(x)


class TestFeedForward:
    """Tests for feed-forward block."""

    def test_output_shape(self) -> None:
        batch, seq_len, d_model = 4, 16, 64
        x = torch.randn(batch, seq_len, d_model)
        ffn = FeedForward(d_model)
        output = ffn(x)
        assert output.shape == x.shape

    def test_numerical_stability(self) -> None:
        ffn = FeedForward(128)
        x = torch.randn(2, 32, 128) * 10
        output = ffn(x)
        assert not torch.isnan(output).any()
        assert not torch.isinf(output).any()

    def test_gradient_flows(self) -> None:
        ffn = FeedForward(64)
        x = torch.randn(2, 8, 64, requires_grad=True)
        output = ffn(x)
        output.sum().backward()
        assert x.grad is not None

    def test_activation_variants(self) -> None:
        """relu and gelu should both produce valid output shapes."""
        x = torch.randn(2, 8, 64)
        for act in ("relu", "gelu"):
            out = FeedForward(64, activation=act)(x)
            assert out.shape == x.shape
            assert not torch.isnan(out).any()

    def test_unknown_activation_raises(self) -> None:
        import pytest
        with pytest.raises(ValueError, match="Unknown activation"):
            FeedForward(64, activation="tanh")


class TestTransformerLayer:
    """Tests for complete transformer layer."""

    def test_output_shape(self) -> None:
        batch, seq_len, d_model, n_heads = 4, 16, 64, 4
        x = torch.randn(batch, seq_len, d_model)
        layer = TransformerLayer(d_model, n_heads)
        output, attn_weights = layer(x)
        assert output.shape == (batch, seq_len, d_model)
        assert attn_weights.shape == (batch, n_heads, seq_len, seq_len)

    def test_numerical_stability(self) -> None:
        """No NaN/Inf on random input."""
        d_model, n_heads = 128, 8
        layer = TransformerLayer(d_model, n_heads)
        for scale in [0.01, 1.0, 10.0]:
            x = torch.randn(2, 32, d_model) * scale
            output, weights = layer(x)
            assert not torch.isnan(output).any(), f"NaN at scale={scale}"
            assert not torch.isinf(output).any(), f"Inf at scale={scale}"

    def test_residual_connection(self) -> None:
        """Output magnitude should be comparable to input (residual keeps scale)."""
        d_model, n_heads = 64, 4
        layer = TransformerLayer(d_model, n_heads)
        x = torch.randn(2, 16, d_model)
        output, _ = layer(x)
        input_std = x.std().item()
        output_std = output.std().item()
        assert output_std < input_std * 3, (
            f"Output std {output_std:.3f} >> input std {input_std:.3f}: possible explosion"
        )
        assert output_std > input_std * 0.1, (
            f"Output std {output_std:.3f} << input std {input_std:.3f}: possible vanishing"
        )

    def test_gradient_flows(self) -> None:
        """Gradients should flow through the entire layer."""
        d_model, n_heads = 64, 4
        layer = TransformerLayer(d_model, n_heads)
        x = torch.randn(2, 8, d_model, requires_grad=True)
        output, _ = layer(x)
        output.sum().backward()
        assert x.grad is not None
        assert not torch.isnan(x.grad).any()
        for name, param in layer.named_parameters():
            assert param.grad is not None, f"No gradient for {name}"

    def test_pre_norm_stability(self) -> None:
        """Pre-norm should keep activations bounded across multiple layers."""
        d_model, n_heads = 64, 4
        layers = torch.nn.ModuleList([
            TransformerLayer(d_model, n_heads) for _ in range(4)
        ])
        x = torch.randn(2, 16, d_model)
        for i, layer in enumerate(layers):
            x, _ = layer(x)
            assert not torch.isnan(x).any(), f"NaN at layer {i}"
            assert x.abs().max() < 100, f"Activation explosion at layer {i}"
