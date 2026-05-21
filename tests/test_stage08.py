"""
Stage 8 tests — Vectorized Surprise Gate.

Tests for:
- VectorizedGate: per-channel-group surprise
- EMA grouped mode
- perception_error_groups
- Integration with model forward_with_cache
"""

import pytest
import torch

from dream_lm.core.surprise_gate import VectorizedGate
from dream_lm.core.ema import EMAStats
from dream_lm.core.predictive_coding import perception_error_groups
from dream_lm.core.model import DREAMLM


# ============================================================
# VectorizedGate tests
# ============================================================

class TestVectorizedGateInit:
    def test_default_init(self):
        gate = VectorizedGate(G=8)
        assert gate.G == 8
        assert gate.theta_0.shape == (8,)
        assert gate.beta.shape == (8,)
        assert (gate.theta_0 == 0.0).all()
        assert (gate.beta == 5.0).all()

    def test_custom_scalar_params(self):
        gate = VectorizedGate(G=4, theta_0=0.5, beta=10.0)
        assert (gate.theta_0 == 0.5).all()
        assert (gate.beta == 10.0).all()

    def test_custom_tensor_params(self):
        theta = torch.tensor([0.0, 0.5, 1.0, 1.5])
        beta = torch.tensor([1.0, 2.0, 3.0, 4.0])
        gate = VectorizedGate(G=4, theta_0=theta, beta=beta)
        assert torch.allclose(gate.theta_0, theta)
        assert torch.allclose(gate.beta, beta)


class TestVectorizedGateOutput:
    def test_output_shape(self):
        gate = VectorizedGate(G=8)
        eps = torch.randn(2, 16, 8)
        out = gate.forward(eps)
        assert out.shape == (2, 16, 8)

    def test_output_range(self):
        gate = VectorizedGate(G=8)
        eps = torch.randn(4, 32, 8) * 10  # wide range
        out = gate.forward(eps)
        assert (out >= 0.0).all()
        assert (out <= 1.0).all()

    def test_per_channel_beta(self):
        """Higher beta produces sharper transitions."""
        theta = torch.zeros(4)
        beta_low = torch.tensor([1.0, 1.0, 1.0, 1.0])
        beta_high = torch.tensor([20.0, 20.0, 20.0, 20.0])

        gate_low = VectorizedGate(G=4, theta_0=theta, beta=beta_low)
        gate_high = VectorizedGate(G=4, theta_0=theta, beta=beta_high)

        eps = torch.ones(1, 1, 4) * 0.5  # above threshold
        out_low = gate_low.forward(eps)
        out_high = gate_high.forward(eps)

        # Higher beta → closer to 1.0 for positive input
        assert (out_high > out_low).all()


class TestVectorizedGateThreshold:
    def test_threshold_effect(self):
        """Higher threshold suppresses gate activation."""
        gate_low = VectorizedGate(G=4, theta_0=0.0)
        gate_high = VectorizedGate(G=4, theta_0=2.0)

        eps = torch.ones(1, 1, 4) * 1.0
        out_low = gate_low.forward(eps)
        out_high = gate_high.forward(eps)

        # Lower threshold → higher gate (input is above threshold)
        assert (out_low > out_high).all()

    def test_threshold_extensibility(self):
        """compute_threshold returns tensor, can be extended."""
        gate = VectorizedGate(G=8, theta_0=0.5)
        threshold = gate.compute_threshold()
        assert isinstance(threshold, torch.Tensor)
        assert threshold.shape == (8,)
        assert (threshold == 0.5).all()


# ============================================================
# EMA grouped mode tests
# ============================================================

class TestEMAGroupedMode:
    def test_is_grouped_property(self):
        scalar = EMAStats.init(batch=2, G=None)
        grouped = EMAStats.init(batch=2, G=8)
        assert not scalar.is_grouped
        assert grouped.is_grouped

    def test_grouped_init_shape(self):
        ema = EMAStats.init(batch=4, G=8)
        assert ema.mu.shape == (4, 8)
        assert ema.var.shape == (4, 8)

    def test_grouped_update_shape(self):
        ema = EMAStats.init(batch=2, G=8)
        eps = torch.abs(torch.randn(2, 16, 8))
        out = ema.update(eps)
        assert out.shape == (2, 16, 8)

    def test_grouped_constant_convergence(self):
        """Constant input converges to zero normalized value."""
        ema = EMAStats.init(batch=1, G=4)
        constant = torch.ones(1, 100, 4) * 5.0
        out = ema.update(constant)
        # After warmup, normalized should trend toward 0
        first_mean = out[:, :10, :].abs().mean()
        last_mean = out[:, -10:, :].abs().mean()
        assert last_mean < first_mean

    def test_grouped_batch_independence(self):
        """Different batch elements have independent statistics."""
        ema = EMAStats.init(batch=2, G=4)
        # Use varying inputs — EMA is scale-invariant for constants
        torch.manual_seed(42)
        eps = torch.abs(torch.randn(2, 50, 4))
        # Add spike to batch 1 only
        eps[1, 25, 0] = 50.0
        out = ema.update(eps)
        # Batch 0 and 1 should have different stats at spike position
        assert not torch.allclose(out[0, 25, :], out[1, 25, :], atol=0.5)
        # mu and var should differ between batches
        assert not torch.allclose(ema.mu[0, :], ema.mu[1, :], atol=0.01)


# ============================================================
# perception_error_groups tests
# ============================================================

class TestPerceptionErrorGroups:
    def test_output_shape(self):
        eps = torch.randn(2, 16, 128)
        out = perception_error_groups(eps, G=8)
        assert out.shape == (2, 16, 8)

    def test_divisible_assertion(self):
        """d_model must be divisible by G."""
        eps = torch.randn(1, 1, 128)
        with pytest.raises(AssertionError):
            perception_error_groups(eps, G=7)

    def test_mean_absolute(self):
        """Output is mean of absolute values per group."""
        eps = torch.ones(1, 1, 16) * 3.0
        out = perception_error_groups(eps, G=4)
        # Each group has 4 channels, mean(|3|) = 3.0
        assert torch.allclose(out, torch.ones(1, 1, 4) * 3.0)

    def test_contiguous_grouping(self):
        """Groups are contiguous chunks of d_model."""
        eps = torch.zeros(1, 1, 16)
        eps[0, 0, 0:4] = 10.0  # first group
        eps[0, 0, 4:8] = 1.0   # second group
        out = perception_error_groups(eps, G=4)
        assert out[0, 0, 0] > out[0, 0, 1]  # first group > second


# ============================================================
# Integration tests
# ============================================================

class TestModelGroupedGate:
    def test_forward_with_grouped_gate(self):
        """Model returns 5-tuple when G > 0."""
        model = DREAMLM(
            vocab_size=100, d_model=128, n_heads=4, n_layers=2, G=8
        )
        x = torch.randint(0, 100, (2, 16))
        out = model(x)
        assert out.logits.shape == (2, 16, 100)

    def test_forward_with_cache_5_tuple(self):
        """forward_with_cache returns 5 values."""
        model = DREAMLM(
            vocab_size=100, d_model=128, n_heads=4, n_layers=2, G=8
        )
        x = torch.randn(2, 8, 128)
        kv_caches = [None] * 2
        ema_stats = EMAStats.init(batch=2)
        ema_grouped = EMAStats.init(batch=2, G=8)

        result = model.forward_with_cache(x, kv_caches, ema_stats, ema_grouped)
        assert len(result) == 5
        output, caches, ema, surprise_scalar, surprise_grouped = result
        assert surprise_scalar.shape == (2, 8)
        assert surprise_grouped.shape == (2, 8, 8)

    def test_backward_compat_G_zero(self):
        """When G=0, surprise_grouped is None."""
        model = DREAMLM(
            vocab_size=100, d_model=128, n_heads=4, n_layers=2, G=0
        )
        x = torch.randn(1, 8, 128)
        kv_caches = [None] * 2
        ema_stats = EMAStats.init(batch=1)

        _, _, _, s_scalar, s_grouped = model.forward_with_cache(
            x, kv_caches, ema_stats
        )
        assert s_scalar is not None
        assert s_grouped is None

    def test_vec_gate_not_created_when_G_zero(self):
        model = DREAMLM(vocab_size=100, d_model=128, G=0)
        assert model.vec_gate is None
        assert model.G == 0

    def test_vec_gate_created_when_G_positive(self):
        model = DREAMLM(vocab_size=100, d_model=128, G=8)
        assert model.vec_gate is not None
        assert model.G == 8
        assert model.vec_gate.G == 8
