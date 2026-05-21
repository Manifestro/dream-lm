"""
Stage 7 — Surprise Gate (Scalar).

Tests for the first plasticity gate mechanism:
    s_t = σ(β·(‖ε_t‖_norm - θ_t))

Coverage:
  - SurpriseGate init and defaults
  - Output range [0, 1]
  - Threshold effect (higher θ → lower s_t)
  - Beta effect (higher β → sharper transition)
  - Gate responds to spike in normalized error
  - compute_threshold extensibility (Stage 8 hook)
  - Integration with model forward (via forward_with_cache)
"""

import torch

from dream_lm.core.surprise_gate import SurpriseGate


# ---------------------------------------------------------------------------
# SurpriseGate basics
# ---------------------------------------------------------------------------
class TestSurpriseGateInit:
    def test_defaults(self):
        """Default theta_0=0.0, beta=5.0."""
        gate = SurpriseGate()
        assert gate.theta_0 == 0.0
        assert gate.beta == 5.0

    def test_custom_values(self):
        """Custom theta_0 and beta should be stored."""
        gate = SurpriseGate(theta_0=1.5, beta=10.0)
        assert gate.theta_0 == 1.5
        assert gate.beta == 10.0


# ---------------------------------------------------------------------------
# Output range
# ---------------------------------------------------------------------------
class TestGateOutputRange:
    def test_range_always_01(self):
        """Gate output should always be in [0, 1]."""
        gate = SurpriseGate()

        # Extreme values
        for eps_values in [
            torch.randn(2, 50) * 100,       # large random
            torch.ones(2, 50) * 1000,       # very positive
            torch.ones(2, 50) * -1000,      # very negative
            torch.zeros(2, 50),             # zero
        ]:
            s = gate.forward(eps_values)
            assert (s >= 0.0).all(), f"Gate output below 0: min={s.min()}"
            assert (s <= 1.0).all(), f"Gate output above 1: max={s.max()}"

    def test_shape_preserved(self):
        """Output shape should match input shape."""
        gate = SurpriseGate()
        for shape in [(1, 10), (4, 64), (8, 128)]:
            eps = torch.randn(*shape)
            s = gate.forward(eps)
            assert s.shape == eps.shape


# ---------------------------------------------------------------------------
# Threshold effect
# ---------------------------------------------------------------------------
class TestThresholdEffect:
    def test_higher_threshold_lowers_gate(self):
        """Higher θ_0 should produce lower s_t for same input."""
        eps = torch.ones(1, 10) * 2.0

        gate_low = SurpriseGate(theta_0=0.0, beta=5.0)
        gate_high = SurpriseGate(theta_0=3.0, beta=5.0)

        s_low = gate_low.forward(eps)
        s_high = gate_high.forward(eps)

        assert (s_high < s_low).all(), (
            f"Higher threshold should lower gate: low={s_low[0,0]:.4f}, high={s_high[0,0]:.4f}"
        )

    def test_zero_threshold_centered(self):
        """With θ_0=0, input=0 should give s_t=0.5 (sigmoid center)."""
        gate = SurpriseGate(theta_0=0.0, beta=5.0)
        eps = torch.zeros(1, 10)
        s = gate.forward(eps)
        assert torch.allclose(s, torch.ones_like(s) * 0.5, atol=1e-6)


# ---------------------------------------------------------------------------
# Beta effect
# ---------------------------------------------------------------------------
class TestBetaEffect:
    def test_higher_beta_sharper(self):
        """Higher β should produce sharper transition (closer to step function)."""
        eps = torch.linspace(-3, 3, 100).unsqueeze(0)  # (1, 100)

        gate_soft = SurpriseGate(theta_0=0.0, beta=1.0)
        gate_hard = SurpriseGate(theta_0=0.0, beta=20.0)

        s_soft = gate_soft.forward(eps)
        s_hard = gate_hard.forward(eps)

        # Hard gate should have steeper gradient around threshold
        grad_soft = s_soft.diff().abs().max()
        grad_hard = s_hard.diff().abs().max()
        assert grad_hard > grad_soft, (
            f"Higher beta should be sharper: soft={grad_soft:.4f}, hard={grad_hard:.4f}"
        )

    def test_beta_very_large_approaches_step(self):
        """Very large β should approach a step function."""
        gate = SurpriseGate(theta_0=0.0, beta=100.0)

        eps_below = torch.tensor([[-1.0]])
        eps_above = torch.tensor([[1.0]])

        s_below = gate.forward(eps_below)
        s_above = gate.forward(eps_above)

        assert s_below.item() < 0.01, f"Below threshold should be ~0: {s_below.item()}"
        assert s_above.item() > 0.99, f"Above threshold should be ~1: {s_above.item()}"


# ---------------------------------------------------------------------------
# Gate responds to spike
# ---------------------------------------------------------------------------
class TestGateRespondsToSpike:
    def test_spike_produces_high_gate(self):
        """A spike in ‖ε‖_norm should produce s_t close to 1."""
        gate = SurpriseGate(theta_0=0.0, beta=5.0)

        eps = torch.zeros(1, 50)
        eps[0, 25] = 5.0  # spike

        s = gate.forward(eps)

        # Baseline (zero input, θ_0=0): s ≈ 0.5
        baseline = s[0, 0].item()
        # Spike: s should be much higher
        spike = s[0, 25].item()

        assert spike > baseline + 0.4, (
            f"Spike should raise gate significantly: baseline={baseline:.4f}, spike={spike:.4f}"
        )

    def test_spike_location_detected(self):
        """The maximum gate value should be at the spike position."""
        gate = SurpriseGate(theta_0=0.0, beta=5.0)

        eps = torch.zeros(1, 50)
        eps[0, 30] = 8.0

        s = gate.forward(eps)
        max_idx = s.argmax().item()

        assert max_idx == 30, f"Max gate at position {max_idx}, expected 30"


# ---------------------------------------------------------------------------
# compute_threshold extensibility
# ---------------------------------------------------------------------------
class TestThresholdExtensibility:
    def test_default_returns_theta_0(self):
        """compute_threshold() with no args should return theta_0."""
        gate = SurpriseGate(theta_0=2.5)
        assert gate.compute_threshold() == 2.5

    def test_token_history_ignored_in_stage_7(self):
        """Passing token_history in Stage 7 should still return theta_0."""
        gate = SurpriseGate(theta_0=1.0)
        token_history = torch.randint(0, 100, (1, 20))

        # Stage 7 ignores token_history — reserved for Stage 8
        assert gate.compute_threshold(token_history) == 1.0


# ---------------------------------------------------------------------------
# Integration with model
# ---------------------------------------------------------------------------
class TestGateWithModelOutput:
    def test_gate_works_with_ema_normalized(self):
        """Gate should work with EMA-normalized output from model pipeline."""
        from dream_lm.core.ema import EMAStats

        torch.manual_seed(42)
        ema = EMAStats.init(batch=1, alpha=0.99, device="cpu", dtype=torch.float32)
        gate = SurpriseGate(theta_0=0.0, beta=5.0)

        # Simulate perception error norms from model
        eps_norm = torch.ones(1, 30) * 11.3 + torch.randn(1, 30) * 0.01
        normalized = ema.update(eps_norm)
        s = gate.forward(normalized)

        assert s.shape == (1, 30)
        assert (s >= 0.0).all() and (s <= 1.0).all()
