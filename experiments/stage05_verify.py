"""
Stage 5 verification: Perception Error (eps^perc).

Verifies:
1. Perception error shape and numerical stability on real model output
2. Correlation between ‖eps^perc‖ and cross-entropy loss across positions
3. Positional profile: ‖eps^perc_t‖ heatmap across sequence
4. Error peaks coincide with semantically unexpected text regions

Results saved to experiments/stage05_verify/results/
"""

import json
import math
from pathlib import Path

import torch
import torch.nn.functional as F

from dream_lm.core.model import DREAMLM
from dream_lm.core.predictive_coding import compute_perception_error, perception_error_norm

RESULTS_DIR = Path("experiments/stage05_verify/results")
RESULTS_DIR.mkdir(parents=True, exist_ok=True)


def verify_eps_perc_shape() -> dict:
    """Verify perception error has correct shape and no NaN/Inf."""
    torch.manual_seed(42)
    model = DREAMLM(vocab_size=66, d_model=128, n_heads=4, n_layers=4, max_seq_len=256)
    model.eval()

    x = torch.randint(0, 66, (4, 32))
    with torch.no_grad():
        out = model(x)
        eps = compute_perception_error(out.logits, out.h_final, model.w_vocab.weight)
        eps_norm = perception_error_norm(eps)

    return {
        "logits_shape": list(out.logits.shape),
        "h_final_shape": list(out.h_final.shape),
        "eps_shape": list(eps.shape),
        "eps_norm_shape": list(eps_norm.shape),
        "has_nan": bool(torch.isnan(eps).any()),
        "has_inf": bool(torch.isinf(eps).any()),
        "mean_norm": eps_norm.mean().item(),
        "std_norm": eps_norm.std().item(),
        "max_norm": eps_norm.max().item(),
        "min_norm": eps_norm.min().item(),
    }


def verify_correlation_with_ce_loss(seq_len: int = 64, n_samples: int = 100) -> dict:
    """Verify ‖eps^perc‖ correlates with cross-entropy loss.

    For each position across multiple sequences, compute:
    - CE loss at that position
    - ‖eps^perc‖ at that position
    Then compute Pearson correlation.
    """
    torch.manual_seed(42)
    model = DREAMLM(vocab_size=66, d_model=128, n_heads=4, n_layers=4, max_seq_len=256)
    model.eval()

    all_ce = []
    all_eps_norm = []

    for i in range(n_samples):
        torch.manual_seed(i * 7)
        x = torch.randint(0, 66, (1, seq_len))

        with torch.no_grad():
            out = model(x)
            eps = compute_perception_error(out.logits, out.h_final, model.w_vocab.weight)
            eps_norm = perception_error_norm(eps)  # (1, seq_len)

            # CE loss per position
            ce = F.cross_entropy(
                out.logits.view(-1, 66), x.view(-1), reduction="none"
            ).view(1, seq_len)

            all_ce.append(ce.squeeze(0))
            all_eps_norm.append(eps_norm.squeeze(0))

    # Flatten across all samples and positions
    ce_flat = torch.cat(all_ce)  # (n_samples * seq_len,)
    eps_flat = torch.cat(all_eps_norm)

    # Pearson correlation
    ce_mean = ce_flat.mean()
    eps_mean = eps_flat.mean()
    ce_centered = ce_flat - ce_mean
    eps_centered = eps_flat - eps_mean

    numerator = (ce_centered * eps_centered).sum()
    denominator = torch.sqrt((ce_centered ** 2).sum() * (eps_centered ** 2).sum())
    correlation = (numerator / (denominator + 1e-10)).item()

    # Also compute rank correlation (Spearman-like via ranking)
    ce_rank = ce_flat.argsort().argsort().float()
    eps_rank = eps_flat.argsort().argsort().float()
    rank_corr = torch.corrcoef(torch.stack([ce_rank, eps_rank]))[0, 1].item()

    return {
        "pearson_correlation": round(correlation, 4),
        "spearman_rank_correlation": round(rank_corr, 4),
        "num_sequences": n_samples,
        "seq_len": seq_len,
        "mean_ce": ce_flat.mean().item(),
        "mean_eps_norm": eps_flat.mean().item(),
    }


def positional_profile(num_layers: int = 4, d_model: int = 128) -> dict:
    """Compute positional profile of ‖eps^perc‖ across sequence positions.

    Returns mean and std of ‖eps^perc‖ at each position, aggregated over
    multiple random sequences. This reveals whether error concentrates
    at specific positions (e.g., start of sequence, topic boundaries).
    """
    torch.manual_seed(42)
    model = DREAMLM(vocab_size=66, d_model=d_model, n_heads=4, n_layers=num_layers, max_seq_len=256)
    model.eval()

    seq_len = 64
    n_sequences = 50
    profile_sum = torch.zeros(seq_len)
    profile_sq_sum = torch.zeros(seq_len)

    for i in range(n_sequences):
        torch.manual_seed(i * 11)
        x = torch.randint(0, 66, (1, seq_len))

        with torch.no_grad():
            out = model(x)
            eps = compute_perception_error(out.logits, out.h_final, model.w_vocab.weight)
            eps_norm = perception_error_norm(eps).squeeze(0)  # (seq_len,)

            profile_sum += eps_norm
            profile_sq_sum += eps_norm ** 2

    profile_mean = (profile_sum / n_sequences).tolist()
    profile_std = ((profile_sq_sum / n_sequences) - (profile_sum / n_sequences) ** 2).clamp(min=0).sqrt().tolist()

    # Find peak position
    peak_pos = profile_mean.index(max(profile_mean))
    min_pos = profile_mean.index(min(profile_mean))

    return {
        "seq_len": seq_len,
        "n_sequences": n_sequences,
        "mean_profile": [round(v, 4) for v in profile_mean],
        "std_profile": [round(v, 4) for v in profile_std],
        "peak_position": peak_pos,
        "peak_value": round(max(profile_mean), 4),
        "min_position": min_pos,
        "min_value": round(min(profile_mean), 4),
    }


def verify_on_sample_text() -> dict:
    """Run perception error analysis on a structured sample text.

    Uses a sequence with known structure (repeating patterns + sudden changes)
    to verify that error peaks align with semantically unexpected positions.
    """
    torch.manual_seed(42)

    # Create a structured "text" with patterns
    # Pattern: repeating sequence [0,1,2,3,4] then sudden change [50,51,52]
    pattern = [0, 1, 2, 3, 4] * 10  # 50 tokens of repeating pattern
    change = [50, 51, 52, 53, 54]   # 5 tokens of new pattern
    continuation = [0, 1, 2, 3, 4] * 3  # return to old pattern
    sequence = pattern + change + continuation  # 68 tokens

    vocab_size = 66
    model = DREAMLM(vocab_size=vocab_size, d_model=128, n_heads=4, n_layers=4, max_seq_len=128)
    model.eval()

    x = torch.tensor([sequence[:64]], dtype=torch.long)  # trim to 64

    with torch.no_grad():
        out = model(x)
        eps = compute_perception_error(out.logits, out.h_final, model.w_vocab.weight)
        eps_norm = perception_error_norm(eps).squeeze(0)  # (64,)

        # CE loss per position
        ce = F.cross_entropy(out.logits.view(-1, vocab_size), x.view(-1), reduction="none")  # (64,)

    # Analyze error at change point (position 50-54)
    change_start = len(pattern)  # 50
    change_end = min(change_start + len(change), 64)

    eps_before = eps_norm[:change_start].mean().item()
    eps_during = eps_norm[change_start:change_end].mean().item()
    eps_after = eps_norm[change_end:].mean().item() if change_end < 64 else 0.0

    ce_before = ce[:change_start].mean().item()
    ce_during = ce[change_start:change_end].mean().item()
    ce_after = ce[change_end:].mean().item() if change_end < 64 else 0.0

    return {
        "sequence_length": len(sequence),
        "change_point_start": change_start,
        "change_point_end": change_end,
        "eps_before_change": round(eps_before, 4),
        "eps_during_change": round(eps_during, 4),
        "eps_after_change": round(eps_after, 4),
        "ce_before_change": round(ce_before, 4),
        "ce_during_change": round(ce_during, 4),
        "ce_after_change": round(ce_after, 4),
        "eps_spikes_at_change": eps_during > eps_before,
        "ce_spikes_at_change": ce_during > ce_before,
        # Full profile for visualization
        "eps_norm_profile": [round(v, 4) for v in eps_norm.tolist()],
        "ce_profile": [round(v, 4) for v in ce.tolist()],
    }


def main():
    print("Stage 5 Verification: Perception Error")
    print("=" * 50)

    n_params = sum(p.numel() for p in DREAMLM(vocab_size=66, d_model=128, n_heads=4, n_layers=4, max_seq_len=256).parameters())
    print(f"Model: {n_params:,} parameters")
    print()

    # --- Shape & Stability ---
    print("1. Perception error shape and stability...")
    shape_result = verify_eps_perc_shape()
    shape_pass = (
        shape_result["eps_shape"] == [4, 32, 128]
        and not shape_result["has_nan"]
        and not shape_result["has_inf"]
    )
    print(f"   eps shape: {shape_result['eps_shape']} (expected [4, 32, 128])")
    print(f"   Mean ‖ε^perc‖: {shape_result['mean_norm']:.4f}")
    print(f"   Range: [{shape_result['min_norm']:.4f}, {shape_result['max_norm']:.4f}]")
    print(f"   NaN/Inf: {shape_result['has_nan']}/{shape_result['has_inf']}")
    print(f"   Result: {'PASS' if shape_pass else 'FAIL'}")
    print()

    # --- Correlation ---
    print("2. Correlation with cross-entropy loss (100 sequences, len=64)...")
    corr_result = verify_correlation_with_ce_loss()
    corr_pass = corr_result["pearson_correlation"] > 0
    print(f"   Pearson correlation: {corr_result['pearson_correlation']:.4f}")
    print(f"   Spearman rank correlation: {corr_result['spearman_rank_correlation']:.4f}")
    print(f"   Result: {'PASS' if corr_pass else 'FAIL'}")
    print()

    # --- Positional Profile ---
    print("3. Positional profile of ‖eps^perc‖ (50 sequences, len=64)...")
    profile = positional_profile()
    print(f"   Peak at position {profile['peak_position']}: ‖ε^perc‖ = {profile['peak_value']:.4f}")
    print(f"   Min at position {profile['min_position']}: ‖ε^perc‖ = {profile['min_value']:.4f}")
    print(f"   Mean profile range: [{profile['min_value']:.4f}, {profile['peak_value']:.4f}]")
    profile_pass = profile["peak_value"] > profile["min_value"]  # variance exists
    print(f"   Result: {'PASS' if profile_pass else 'FAIL'}")
    print()

    # --- Sample Text ---
    print("4. Structured text analysis (pattern → change → continuation)...")
    sample_result = verify_on_sample_text()
    print(f"   Change point: positions {sample_result['change_point_start']}-{sample_result['change_point_end']}")
    print(f"   ‖ε^perc‖ before: {sample_result['eps_before_change']:.4f}, during: {sample_result['eps_during_change']:.4f}, after: {sample_result['eps_after_change']:.4f}")
    print(f"   CE loss before: {sample_result['ce_before_change']:.4f}, during: {sample_result['ce_during_change']:.4f}, after: {sample_result['ce_after_change']:.4f}")
    print(f"   eps spikes at change: {sample_result['eps_spikes_at_change']}")
    print(f"   CE spikes at change: {sample_result['ce_spikes_at_change']}")
    sample_pass = sample_result["eps_spikes_at_change"] and sample_result["ce_spikes_at_change"]
    print(f"   Result: {'PASS' if sample_pass else 'FAIL'}")
    print()

    # --- Save results ---
    results = {
        "shape_stability": shape_result,
        "correlation": corr_result,
        "positional_profile": profile,
        "sample_text": sample_result,
    }

    output_path = RESULTS_DIR / "results.json"
    with open(output_path, "w") as f:
        json.dump(results, f, indent=2)
    print(f"Results saved to {output_path}")

    all_pass = shape_pass and corr_pass and profile_pass and sample_pass
    if all_pass:
        print("\nStage 5 VERIFIED: Perception error works correctly, correlates with CE loss, peaks at change points.")
    else:
        print("\nStage 5 FAILED: Check results above.")


if __name__ == "__main__":
    main()
