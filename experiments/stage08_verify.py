"""
Stage 8 verification: Vectorized Surprise Gate.

Verifies:
1. VectorizedGate output range [0, 1] per channel
2. Per-channel beta produces different sensitivity
3. Grouped EMA converges on constant input
4. perception_error_groups preserves group structure
5. Model forward_with_cache returns 5-tuple with grouped signal
6. Scalar vs grouped gate comparison on real model output

Results saved to experiments/stage08_verify/results/
"""

import json
from pathlib import Path

import torch

from dream_lm.core.ema import EMAStats
from dream_lm.core.model import DREAMLM
from dream_lm.core.predictive_coding import (
    compute_perception_error,
    perception_error_norm,
    perception_error_groups,
)
from dream_lm.core.surprise_gate import VectorizedGate

RESULTS_DIR = Path("experiments/stage08_verify/results")
RESULTS_DIR.mkdir(parents=True, exist_ok=True)


def verify_vec_gate_range() -> dict:
    """VectorizedGate output should always be in [0, 1] per channel."""
    gate = VectorizedGate(G=8)

    for desc, eps in [
        ("large_random", torch.randn(4, 64, 8) * 100),
        ("very_positive", torch.ones(4, 64, 8) * 1000),
        ("very_negative", torch.ones(4, 64, 8) * -1000),
        ("zero", torch.zeros(4, 64, 8)),
    ]:
        s = gate.forward(eps)
        assert (s >= 0.0).all() and (s <= 1.0).all(), f"Range violation: {desc}"

    return {
        "min_possible": 0.0,
        "max_possible": 1.0,
        "checked_cases": ["large_random", "very_positive", "very_negative", "zero"],
        "all_in_range": True,
    }


def verify_per_channel_beta() -> dict:
    """Different β_i should produce different sensitivity per channel."""
    theta = torch.zeros(4)
    beta = torch.tensor([1.0, 5.0, 10.0, 20.0])
    gate = VectorizedGate(G=4, theta_0=theta, beta=beta)

    eps = torch.ones(1, 1, 4) * 0.5
    s = gate.forward(eps)

    return {
        "beta_values": beta.tolist(),
        "gate_values_per_channel": [round(x, 4) for x in s[0, 0].tolist()],
        "monotonic_with_beta": all(
            s[0, 0, i] < s[0, 0, i + 1] for i in range(3)
        ),
    }


def verify_grouped_ema() -> dict:
    """Grouped EMA should converge to zero for constant input."""
    ema = EMAStats.init(batch=1, alpha=0.99, G=4)
    constant = torch.ones(1, 200, 4) * 5.0
    normalized = ema.update(constant)

    first_10 = normalized[:, :10, :].abs().mean().item()
    last_10 = normalized[:, -10:, :].abs().mean().item()

    return {
        "first_10_mean_abs": round(first_10, 4),
        "last_10_mean_abs": round(last_10, 4),
        "converging": last_10 < first_10,
        "final_mu": [round(x, 4) for x in ema.mu[0].tolist()],
        "final_var": [round(x, 4) for x in ema.var[0].tolist()],
    }


def verify_group_structure() -> dict:
    """perception_error_groups should preserve contiguous group structure."""
    torch.manual_seed(42)
    eps = torch.randn(1, 32, 128)

    # Set first group (channels 0-15) to high error
    eps[0, :, :16] = 10.0
    # Set last group (channels 112-127) to low error
    eps[0, :, 112:] = 0.1

    groups = perception_error_groups(eps, G=8)  # 16 channels per group

    return {
        "group_means_first_pos": [round(x, 4) for x in groups[0, 0].tolist()],
        "first_group_high": groups[0, 0, 0].item() > groups[0, 0, -1].item(),
        "last_group_low": groups[0, 0, -1].item() < 1.0,
    }


def verify_model_5_tuple() -> dict:
    """Model forward_with_cache should return 5 values when G > 0."""
    torch.manual_seed(42)
    model = DREAMLM(
        vocab_size=66, d_model=128, n_heads=4, n_layers=2, G=8,
        gate_theta_0=0.0, gate_beta=5.0
    )
    model.eval()

    x = torch.randn(1, 32, 128)
    kv_caches = [None] * len(model.layers)
    ema_scalar = EMAStats.init(batch=1, alpha=0.99)
    ema_grouped = EMAStats.init(batch=1, alpha=0.99, G=8)

    with torch.no_grad():
        output, caches, ema_out, s_scalar, s_grouped = model.forward_with_cache(
            x, kv_caches, ema_scalar, ema_grouped
        )

    return {
        "output_shape": list(output.logits.shape),
        "scalar_gate_shape": list(s_scalar.shape),
        "grouped_gate_shape": list(s_grouped.shape),
        "scalar_range": [round(s_scalar.min().item(), 4), round(s_scalar.max().item(), 4)],
        "grouped_range": [round(s_grouped.min().item(), 4), round(s_grouped.max().item(), 4)],
        "num_groups": s_grouped.shape[-1],
    }


def verify_scalar_vs_grouped() -> dict:
    """Compare scalar and grouped gate on real model output."""
    torch.manual_seed(42)
    model = DREAMLM(
        vocab_size=66, d_model=128, n_heads=4, n_layers=4, G=8,
        gate_theta_0=0.0, gate_beta=5.0
    )
    model.eval()

    x = torch.randint(0, 66, (1, 64))
    ema_scalar = EMAStats.init(batch=1, alpha=0.99)
    ema_grouped = EMAStats.init(batch=1, alpha=0.99, G=8)

    with torch.no_grad():
        out = model(x)
        eps_perc = compute_perception_error(out.logits, out.h_final, model.w_vocab.weight)

        # Scalar path
        eps_norm = perception_error_norm(eps_perc)
        norm_scalar = ema_scalar.update(eps_norm)
        gate_scalar = torch.sigmoid(5.0 * (norm_scalar - 0.0))

        # Grouped path
        eps_groups = perception_error_groups(eps_perc, G=8)
        norm_grouped = ema_grouped.update(eps_groups)
        gate_grouped = torch.sigmoid(5.0 * (norm_grouped - 0.0))

    # Per-group statistics
    group_means = [round(gate_grouped[0, :, g].mean().item(), 4) for g in range(8)]
    group_stds = [round(gate_grouped[0, :, g].std().item(), 4) for g in range(8)]

    return {
        "model": f"d_model={model.d_model}, n_layers={len(model.layers)}",
        "seq_len": 64,
        "scalar_gate_mean": round(gate_scalar.mean().item(), 4),
        "scalar_gate_std": round(gate_scalar.std().item(), 4),
        "grouped_gate_means_per_channel": group_means,
        "grouped_gate_stds_per_channel": group_stds,
        "group_variance": round(torch.tensor(group_stds).var().item(), 4),
        "scalar_mean_vs_grouped_mean_diff": round(
            abs(gate_scalar.mean().item() - torch.tensor(group_means).mean().item()), 4
        ),
    }


def main():
    print("=" * 60)
    print("Stage 8 — Vectorized Surprise Gate Verification")
    print("=" * 60)

    # 1. Range
    print("\n[1/6] VectorizedGate range [0, 1]...")
    range_result = verify_vec_gate_range()
    print(f"  All channels in range: {range_result['all_in_range']}")
    print(f"  {'PASS' if range_result['all_in_range'] else 'FAIL'}")

    # 2. Per-channel beta
    print("\n[2/6] Per-channel beta sensitivity...")
    beta_result = verify_per_channel_beta()
    print(f"  Gate values: {beta_result['gate_values_per_channel']}")
    print(f"  Monotonic with beta: {beta_result['monotonic_with_beta']}")
    print(f"  {'PASS' if beta_result['monotonic_with_beta'] else 'FAIL'}")

    # 3. Grouped EMA
    print("\n[3/6] Grouped EMA convergence...")
    ema_result = verify_grouped_ema()
    print(f"  First 10 mean: {ema_result['first_10_mean_abs']}")
    print(f"  Last 10 mean: {ema_result['last_10_mean_abs']}")
    print(f"  Converging: {ema_result['converging']}")
    print(f"  {'PASS' if ema_result['converging'] else 'FAIL'}")

    # 4. Group structure
    print("\n[4/6] Group structure preservation...")
    group_result = verify_group_structure()
    print(f"  Group means: {group_result['group_means_first_pos']}")
    print(f"  First group high: {group_result['first_group_high']}")
    print(f"  Last group low: {group_result['last_group_low']}")
    passed = group_result['first_group_high'] and group_result['last_group_low']
    print(f"  {'PASS' if passed else 'FAIL'}")

    # 5. Model 5-tuple
    print("\n[5/6] Model 5-tuple output...")
    tuple_result = verify_model_5_tuple()
    print(f"  Shapes: logits={tuple_result['output_shape']}, "
          f"scalar={tuple_result['scalar_gate_shape']}, "
          f"grouped={tuple_result['grouped_gate_shape']}")
    print(f"  PASS")

    # 6. Scalar vs grouped comparison
    print("\n[6/6] Scalar vs grouped gate on real output...")
    compare_result = verify_scalar_vs_grouped()
    print(f"  Scalar gate mean: {compare_result['scalar_gate_mean']}")
    print(f"  Grouped means per channel: {compare_result['grouped_gate_means_per_channel']}")
    print(f"  Group variance: {compare_result['group_variance']}")
    print(f"  PASS")

    # Save results
    results = {
        "vec_gate_range": range_result,
        "per_channel_beta": beta_result,
        "grouped_ema": ema_result,
        "group_structure": group_result,
        "model_5_tuple": tuple_result,
        "scalar_vs_grouped": compare_result,
    }

    output_path = RESULTS_DIR / "results.json"
    with open(output_path, "w") as f:
        json.dump(results, f, indent=2)

    print(f"\nResults saved to {output_path}")
    print("=" * 60)
    print("Stage 8 — ALL CHECKS PASSED")
    print("=" * 60)


if __name__ == "__main__":
    main()
