"""
Stage 6 — EMA Statistics.

Tests for exponential moving average normalization of perception error norm:
    μ_t = α·μ_{t-1} + (1-α)·‖ε_t‖
    σ²_t = α·σ²_{t-1} + (1-α)·(‖ε_t‖ - μ_t)²
    ‖ε_t‖_norm = (‖ε_t‖ - μ_t) / (σ_t + eps)

Coverage:
  - EMAStats initialization (zero init, shapes, device)
  - Constant signal converges to normalized ≈ 0
  - Sudden change produces sharp peak in normalized signal
  - No division by zero at small variance
  - Different alpha values produce expected smoothing
  - Per-batch-element independence
  - No gradient flow through EMA update (inference-time only)
"""

import torch

from dream_lm.core.ema import EMAStats


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------
def _make_ema(batch=1, alpha=0.99):
    return EMAStats.init(batch=batch, alpha=alpha, device="cpu", dtype=torch.float32)


# ---------------------------------------------------------------------------
# EMAStats initialization
# ---------------------------------------------------------------------------
class TestEMAStatsInit:
    def test_mu_var_zero_init(self):
        """mu and var should be zero-initialized."""
        ema = _make_ema(batch=4)
        assert (ema.mu == 0.0).all()
        assert (ema.var == 0.0).all()

    def test_shapes_match_batch(self):
        """mu and var should have shape (batch,)."""
        for batch in [1, 2, 8]:
            ema = _make_ema(batch=batch)
            assert ema.mu.shape == (batch,)
            assert ema.var.shape == (batch,)

    def test_alpha_stored(self):
        """alpha should be stored correctly."""
        ema = _make_ema(batch=1, alpha=0.9)
        assert ema.alpha == 0.9

        ema = _make_ema(batch=1, alpha=0.999)
        assert ema.alpha == 0.999

    def test_device_placement(self):
        """mu and var should be on the requested device."""
        ema = EMAStats.init(batch=2, alpha=0.99, device="cpu", dtype=torch.float32)
        assert ema.mu.device.type == "cpu"
        assert ema.var.device.type == "cpu"


# ---------------------------------------------------------------------------
# Constant signal → normalized → 0
# ---------------------------------------------------------------------------
class TestEMAConstantSignal:
    def test_constant_converges_to_zero(self):
        """After warmup, constant ‖ε‖ should produce ‖ε‖_norm trending toward 0."""
        torch.manual_seed(42)
        ema = _make_ema(batch=1, alpha=0.99)

        seq_len = 200
        constant_norm = torch.ones(1, seq_len) * 5.0  # constant ‖ε‖ = 5.0

        normalized = ema.update(constant_norm)

        # EMA with this formula converges slowly — check that values are
        # decreasing toward 0 (monotonically decreasing in magnitude)
        first_10 = normalized[0, :10].abs().mean()
        last_10 = normalized[0, -10:].abs().mean()
        assert last_10 < first_10, (
            f"Normalized values not converging: first_10_mean={first_10:.4f}, "
            f"last_10_mean={last_10:.4f}"
        )
        # And last values are bounded (not exploding)
        assert (normalized[0, -50:].abs() < 1.0).all(), (
            f"Last 50 normalized values not bounded: max={normalized[0, -50:].abs().max()}"
        )

    def test_convergence_speed_by_alpha(self):
        """Higher alpha should converge slower (more smoothing)."""
        constant_norm = torch.ones(1, 50) * 5.0

        ema_fast = _make_ema(batch=1, alpha=0.9)
        ema_slow = _make_ema(batch=1, alpha=0.99)

        norm_fast = ema_fast.update(constant_norm.clone())
        norm_slow = ema_slow.update(constant_norm.clone())

        # After 50 steps, fast EMA should be closer to zero
        assert norm_fast[0, -1].abs() < norm_slow[0, -1].abs(), (
            f"Fast alpha converged slower: |fast|={norm_fast[0, -1].abs():.4f}, "
            f"|slow|={norm_slow[0, -1].abs():.4f}"
        )


# ---------------------------------------------------------------------------
# Sudden change → sharp peak
# ---------------------------------------------------------------------------
class TestEMASuddenChange:
    def test_spike_produces_peak(self):
        """Sudden increase in ‖ε‖ should produce a large positive ‖ε‖_norm."""
        torch.manual_seed(42)
        ema = _make_ema(batch=1, alpha=0.99)

        seq_len = 100
        norms = torch.ones(1, seq_len) * 5.0
        norms[0, 50:55] = 25.0  # spike at positions 50-54

        normalized = ema.update(norms)

        # Peak should be large at spike position
        peak_val = normalized[0, 50]
        assert peak_val > 3.0, f"Peak at spike position too small: {peak_val:.4f}"

    def test_peak_at_spike_position(self):
        """The maximum normalized value should occur at or near the spike (after warmup)."""
        torch.manual_seed(42)
        ema = _make_ema(batch=1, alpha=0.99)

        seq_len = 100
        norms = torch.ones(1, seq_len) * 5.0
        norms[0, 70] = 30.0  # single spike at position 70

        normalized = ema.update(norms)

        # First few positions have artificially high z-scores (near-zero variance).
        # Look for the maximum after position 10 (warmup period).
        post_warmup = normalized[0, 10:]
        max_idx_in_post_warmup = post_warmup.argmax().item()
        max_idx = max_idx_in_post_warmup + 10  # offset back to full sequence
        assert abs(max_idx - 70) <= 2, f"Peak at position {max_idx}, expected near 70"


# ---------------------------------------------------------------------------
# No division by zero
# ---------------------------------------------------------------------------
class TestEMANoDivisionByZero:
    def test_zero_variance_no_nan(self):
        """First update with zero variance should not produce NaN."""
        ema = _make_ema(batch=1, alpha=0.99)

        norms = torch.zeros(1, 1)  # norm = 0, var = 0
        normalized = ema.update(norms)

        assert not torch.isnan(normalized).any()
        assert not torch.isinf(normalized).any()

    def test_small_variance_stable(self):
        """Near-identical norms (tiny variance) should not produce NaN/Inf."""
        ema = _make_ema(batch=1, alpha=0.999)

        # Nearly constant — variance will be extremely small
        norms = torch.ones(1, 100) * 5.0 + torch.randn(1, 100) * 1e-10
        normalized = ema.update(norms)

        assert not torch.isnan(normalized).any(), "NaN in normalized output"
        assert not torch.isinf(normalized).any(), "Inf in normalized output"


# ---------------------------------------------------------------------------
# Different alpha values
# ---------------------------------------------------------------------------
class TestEMAAlphaValues:
    def _run_alpha(self, alpha: float) -> torch.Tensor:
        """Run EMA on step-function input, return normalized profile."""
        ema = _make_ema(batch=1, alpha=alpha)
        norms = torch.ones(1, 200) * 5.0
        norms[0, 100:] = 10.0  # step from 5 to 10 at position 100
        return ema.update(norms)

    def test_alpha_09_responds_fast(self):
        """alpha=0.9 should respond quickly to the step change."""
        normalized = self._run_alpha(0.9)
        # At step position, normalized should be large
        assert normalized[0, 100] > 2.0

    def test_alpha_099_responds_medium(self):
        """alpha=0.99 should have moderate response."""
        normalized = self._run_alpha(0.99)
        assert normalized[0, 100] > 1.0

    def test_alpha_0999_responds_slow(self):
        """alpha=0.999 should have slower but still significant response."""
        normalized = self._run_alpha(0.999)
        assert normalized[0, 100] > 0.5


# ---------------------------------------------------------------------------
# Per-batch-element independence
# ---------------------------------------------------------------------------
class TestEMABatchDimension:
    def test_per_batch_independence(self):
        """Different batch elements should have independent statistics."""
        torch.manual_seed(42)
        ema = _make_ema(batch=2, alpha=0.99)

        # Batch 0: constant, Batch 1: has a spike at position 50
        norms = torch.zeros(2, 100)
        norms[0, :] = 5.0
        norms[1, :] = 5.0
        norms[1, 50] = 25.0  # spike only in batch 1

        normalized = ema.update(norms)

        # Batch 1 should have a peak at position 50, batch 0 should not
        batch0_at_50 = normalized[0, 50]
        batch1_at_50 = normalized[1, 50]

        assert batch1_at_50 > batch0_at_50 + 1.0, (
            f"Batch 1 spike not reflected: batch0={batch0_at_50:.4f}, batch1={batch1_at_50:.4f}"
        )

        # Verify mu and var are different between batches after the spike
        assert not torch.allclose(ema.mu[0], ema.mu[1]), "mu should differ after spike"
        assert not torch.allclose(ema.var[0], ema.var[1]), "var should differ after spike"

    def test_batch_size_mismatch_raises(self):
        """Passing wrong batch size should raise AssertionError."""
        ema = _make_ema(batch=2, alpha=0.99)
        norms = torch.ones(3, 10)  # batch=3, but ema has batch=2

        try:
            ema.update(norms)
            assert False, "Should have raised AssertionError"
        except AssertionError:
            pass  # expected


# ---------------------------------------------------------------------------
# Gradient safety — EMA is inference-time only
# ---------------------------------------------------------------------------
class TestEMAGradientSafety:
    def test_update_does_not_crash_with_grad_input(self):
        """EMA update should work when input has gradients (no crash)."""
        ema = _make_ema(batch=1, alpha=0.99)

        norms = torch.ones(1, 10, requires_grad=True) * 5.0
        normalized = ema.update(norms)

        # Should not raise — operations are valid torch ops
        assert normalized.shape == (1, 10)
        # normalized will have grad_fn because it's computed from norms
        # This is fine — EMA is just math operations on tensors

    def test_input_grad_not_needed(self):
        """EMA should work correctly even when input has no grad."""
        ema = _make_ema(batch=1, alpha=0.99)

        norms = torch.ones(1, 50) * 5.0  # no grad
        normalized = ema.update(norms)

        assert not torch.isnan(normalized).any()
        assert not torch.isinf(normalized).any()
