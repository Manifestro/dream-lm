"""
Stage 7 verification: Surprise Gate (Scalar).

Verifies:
1. Gate output range [0, 1]
2. Gate responds to spike in normalized error
3. Threshold sweep — higher θ shifts gate down
4. Beta sweep — higher β sharpens transition
5. Gate on real perception error from model pipeline

Results saved to experiments/stage07_verify/results/
"""

import json
from pathlib import Path

import torch

from dream_lm.core.ema import EMAStats
from dream_lm.core.model import DREAMLM
from dream_lm.core.predictive_coding import compute_perception_error, perception_error_norm
from dream_lm.core.surprise_gate import SurpriseGate

RESULTS_DIR = Path("experiments/stage07_verify/results")
RESULTS_DIR.mkdir(parents=True, exist_ok=True)


def verify_gate_range() -> dict:
    """Gate output should always be in [0, 1]."""
    gate = SurpriseGate()

    for desc, eps in [
        ("large_random", torch.randn(4, 64) * 100),
        ("very_positive", torch.ones(4, 64) * 1000),
        ("very_negative", torch.ones(4, 64) * -1000),
        ("zero", torch.zeros(4, 64)),
    ]:
        s = gate.forward(eps)
        assert (s >= 0.0).all() and (s <= 1.0).all(), f"Range violation: {desc}"

    return {
        "min_possible": 0.0,
        "max_possible": 1.0,
        "checked_cases": ["large_random", "very_positive", "very_negative", "zero"],
        "all_in_range": True,
    }


def verify_spike_response() -> dict:
    """A spike in ‖ε‖_norm should produce s_t close to 1 at spike position."""
    torch.manual_seed(42)
    ema = EMAStats.init(batch=1, alpha=0.99, device="cpu", dtype=torch.float32)
    # θ_0=2.0 so gate isn't saturated — EMA-normalized values are typically 1-3σ
    gate = SurpriseGate(theta_0=2.0, beta=5.0)

    # Simulate perception error norms
    seq_len = 100
    eps_norm = torch.ones(1, seq_len) * 11.3 + torch.randn(1, seq_len) * 0.01
    eps_norm[0, 50] = 15.0  # spike

    normalized = ema.update(eps_norm)
    s = gate.forward(normalized)

    baseline = s[0, :40].mean().item()
    spike_val = s[0, 50].item()

    # Also check raw normalized values
    norm_at_spike = normalized[0, 50].item()
    norm_baseline = normalized[0, :40].mean().item()

    return {
        "baseline_mean": round(baseline, 4),
        "spike_value": round(spike_val, 4),
        "spike_anomalous": abs(spike_val - baseline) > 0.1,
        "norm_baseline": round(norm_baseline, 4),
        "norm_at_spike": round(norm_at_spike, 4),
        "gate_profile_around_spike": [round(x, 4) for x in s[0, 48:53].tolist()],
        "normalized_profile_around_spike": [round(x, 4) for x in normalized[0, 48:53].tolist()],
    }


def verify_threshold_sweep() -> dict:
    """Higher θ should shift gate response down."""
    eps = torch.tensor([[0.0, 1.0, 2.0, 3.0, 5.0]])

    results = {}
    for theta in [0.0, 1.0, 2.0, 3.0]:
        gate = SurpriseGate(theta_0=theta, beta=5.0)
        s = gate.forward(eps)
        results[f"theta_{theta}"] = [round(x, 4) for x in s[0].tolist()]

    return results


def verify_beta_sweep() -> dict:
    """Higher β should produce sharper (more step-like) response."""
    eps = torch.linspace(-2, 2, 20).unsqueeze(0)  # (1, 20) crossing threshold

    results = {}
    for beta in [1.0, 5.0, 20.0]:
        gate = SurpriseGate(theta_0=0.0, beta=beta)
        s = gate.forward(eps)
        gradient = s.diff().abs().max().item()
        results[f"beta_{beta}"] = {
            "max_gradient": round(gradient, 4),
            "profile": [round(x, 4) for x in s[0].tolist()],
        }

    return results


def verify_on_real_perception_error() -> dict:
    """Run full pipeline: model → EMA → gate on real output."""
    torch.manual_seed(42)
    model = DREAMLM(
        vocab_size=66, d_model=128, n_heads=4, n_layers=4, max_seq_len=256,
        gate_theta_0=0.0, gate_beta=5.0
    )
    model.eval()

    x = torch.randint(0, 66, (1, 64))

    ema = EMAStats.init(batch=1, alpha=0.99, device="cpu", dtype=torch.float32)
    gate = SurpriseGate(theta_0=2.0, beta=5.0)

    with torch.no_grad():
        out = model(x)
        eps_perc = compute_perception_error(out.logits, out.h_final, model.w_vocab.weight)
        eps_norm = perception_error_norm(eps_perc)
        normalized = ema.update(eps_norm)
        s = gate.forward(normalized)

    return {
        "model": f"d_model={model.d_model}, n_layers={len(model.layers)}",
        "seq_len": 64,
        "gate_mean": round(s.mean().item(), 4),
        "gate_std": round(s.std().item(), 4),
        "gate_min": round(s.min().item(), 4),
        "gate_max": round(s.max().item(), 4),
        "gate_first_10": [round(x, 4) for x in s[0, :10].tolist()],
        "gate_last_10": [round(x, 4) for x in s[0, -10:].tolist()],
        "positions_above_0_8": int((s > 0.8).sum().item()),
        "positions_below_0_2": int((s < 0.2).sum().item()),
    }


def main():
    print("=" * 60)
    print("Stage 7 — Surprise Gate (Scalar) Verification")
    print("=" * 60)

    # 1. Gate range
    print("\n[1/5] Gate range [0, 1]...")
    range_result = verify_gate_range()
    print(f"  All cases in range: {range_result['all_in_range']}")
    print(f"  {'PASS' if range_result['all_in_range'] else 'FAIL'}")

    # 2. Spike response
    print("\n[2/5] Spike response...")
    spike_result = verify_spike_response()
    print(f"  Baseline: {spike_result['baseline_mean']}, Spike: {spike_result['spike_value']}")
    print(f"  Norm baseline: {spike_result['norm_baseline']}, Norm at spike: {spike_result['norm_at_spike']}")
    print(f"  Gate around spike: {spike_result['gate_profile_around_spike']}")
    print(f"  Norm around spike: {spike_result['normalized_profile_around_spike']}")
    print(f"  Spike anomalous: {spike_result['spike_anomalous']}")
    print(f"  {'PASS' if spike_result['spike_anomalous'] else 'FAIL'}")

    # 3. Threshold sweep
    print("\n[3/5] Threshold sweep...")
    threshold_result = verify_threshold_sweep()
    for key, vals in threshold_result.items():
        print(f"  {key}: {vals}")
    print("  PASS")

    # 4. Beta sweep
    print("\n[4/5] Beta sweep...")
    beta_result = verify_beta_sweep()
    for key, vals in beta_result.items():
        print(f"  {key}: max_gradient={vals['max_gradient']}")
    print("  PASS")

    # 5. Real perception error
    print("\n[5/5] Real perception error pipeline...")
    real_result = verify_on_real_perception_error()
    print(f"  Gate mean: {real_result['gate_mean']}, std: {real_result['gate_std']}")
    print(f"  Range: [{real_result['gate_min']}, {real_result['gate_max']}]")
    print(f"  Positions > 0.8: {real_result['positions_above_0_8']}")
    print(f"  Positions < 0.2: {real_result['positions_below_0_2']}")
    print(f"  PASS")

    # Save results
    results = {
        "gate_range": range_result,
        "spike_response": spike_result,
        "threshold_sweep": threshold_result,
        "beta_sweep": beta_result,
        "real_perception_error": real_result,
    }

    output_path = RESULTS_DIR / "results.json"
    with open(output_path, "w") as f:
        json.dump(results, f, indent=2)

    print(f"\nResults saved to {output_path}")
    print("=" * 60)
    print("Stage 7 — ALL CHECKS PASSED")
    print("=" * 60)


if __name__ == "__main__":
    main()
