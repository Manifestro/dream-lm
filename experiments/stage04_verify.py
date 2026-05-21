"""
Stage 4 verification: Low-Rank Fast Weights.

Verifies:
1. Baseline identity: fast_weight_r=0 produces bit-identical output to no-fast-weights model
2. Augment verification: non-zero U changes K/V output
3. V_basis orthonormality check
4. Performance: fast weights add minimal overhead at r=16

Results saved to experiments/stage04_verify/results/
"""

import json
import time
from pathlib import Path

import torch

from dream_lm.core.fast_weights import FastWeightState
from dream_lm.core.model import DREAMLM

RESULTS_DIR = Path("experiments/stage04_verify/results")
RESULTS_DIR.mkdir(parents=True, exist_ok=True)


def verify_baseline_identity(n_tests: int = 10) -> bool:
    """Verify that fast_weight_r=0 produces identical output to baseline model."""
    for i in range(n_tests):
        torch.manual_seed(i * 13)

        model_base = DREAMLM(vocab_size=65, d_model=64, n_heads=4, n_layers=2, max_seq_len=128)
        model_fw = DREAMLM(vocab_size=65, d_model=64, n_heads=4, n_layers=2, max_seq_len=128, fast_weight_r=0)

        # Ensure identical weights
        model_fw.load_state_dict(model_base.state_dict())
        model_base.eval()
        model_fw.eval()

        prompt_len = (i % 20) + 1
        prompt = [j % 65 for j in range(prompt_len)]

        with torch.no_grad():
            tokens_base = model_base.generate(prompt, max_new_tokens=5, temperature=0.0)
            tokens_fw = model_fw.generate(prompt, max_new_tokens=5, temperature=0.0)

        if tokens_base != tokens_fw:
            print(f"  FAIL test={i} prompt_len={prompt_len}")
            print(f"    base: {tokens_base}")
            print(f"    fw:   {tokens_fw}")
            return False

    return True


def verify_augment_effect() -> dict:
    """Verify that non-zero fast weights actually change the output."""
    torch.manual_seed(42)
    model_zero = DREAMLM(vocab_size=65, d_model=64, n_heads=4, n_layers=2, max_seq_len=128, fast_weight_r=16)
    model_nonzero = DREAMLM(vocab_size=65, d_model=64, n_heads=4, n_layers=2, max_seq_len=128, fast_weight_r=16)

    # Copy base weights
    model_nonzero.load_state_dict(model_zero.state_dict())
    model_zero.eval()
    model_nonzero.eval()

    # Manually set non-zero fast weights in the model's caches
    # We'll test this at the FastWeightState level
    fw_zero = FastWeightState.init(n_heads=4, d_head=16, r=16)
    fw_nonzero = FastWeightState.init(n_heads=4, d_head=16, r=16)
    fw_nonzero.u_k.fill_(0.1)
    fw_nonzero.u_v.fill_(0.1)

    k = torch.randn(1, 4, 10, 16)
    v = torch.randn(1, 4, 10, 16)

    k_zero, v_zero = fw_zero.augment(k, v)
    k_nz, v_nz = fw_nonzero.augment(k, v)

    k_changed = not torch.allclose(k_zero, k_nz)
    v_changed = not torch.allclose(v_zero, v_nz)

    # Measure relative change
    k_rel_change = (k_nz - k_zero).norm() / k_zero.norm()
    v_rel_change = (v_nz - v_zero).norm() / v_zero.norm()

    return {
        "k_changed": k_changed,
        "v_changed": v_changed,
        "k_relative_change": k_rel_change.item(),
        "v_relative_change": v_rel_change.item(),
    }


def verify_v_basis_orthonormality(n_heads: int = 4, d_head: int = 16, r: int = 8) -> bool:
    """Verify V_basis^T @ V_basis = I for all heads."""
    fw = FastWeightState.init(n_heads=n_heads, d_head=d_head, r=r)

    for h in range(n_heads):
        v_h = fw.v_basis[h]
        gram = v_h.T @ v_h
        if not torch.allclose(gram, torch.eye(r), atol=1e-5):
            print(f"  FAIL head={h}: V_basis^T @ V_basis != I")
            return False

    return True


def benchmark_fast_weights_overhead(n_runs: int = 20, warmup: int = 3) -> dict:
    """Measure ms/token overhead of fast weights at different ranks."""
    ranks = [0, 8, 16, 32]
    results = {}

    for r in ranks:
        torch.manual_seed(42)
        model = DREAMLM(vocab_size=65, d_model=64, n_heads=4, n_layers=2, max_seq_len=128, fast_weight_r=r)
        model.eval()

        prompt = [i % 65 for i in range(32)]

        times = []
        for run in range(warmup + n_runs):
            torch.manual_seed(42)
            t0 = time.perf_counter()
            with torch.no_grad():
                model.generate(prompt, max_new_tokens=10, temperature=0.0)
            elapsed = time.perf_counter() - t0
            if run >= warmup:
                times.append(elapsed)

        avg_ms = (sum(times) / len(times)) * 1000
        results[f"r={r}"] = {
            "total_ms": avg_ms,
            "ms_per_token": avg_ms / 10,
        }

    return results


def main():
    print("Stage 4 Verification: Low-Rank Fast Weights")
    print("=" * 50)

    n_params_base = sum(p.numel() for p in DREAMLM(vocab_size=65, d_model=64, n_heads=4, n_layers=2, max_seq_len=128).parameters())
    print(f"Base model: {n_params_base:,} parameters (fast_weight_r=0)")
    print()

    # --- Baseline Identity ---
    print("Baseline identity verification (10 tests)...")
    identity_pass = verify_baseline_identity(n_tests=10)
    print(f"  Result: {'PASS' if identity_pass else 'FAIL'}")
    print()

    # --- Augment Effect ---
    print("Augment effect verification...")
    augment_result = verify_augment_effect()
    augment_pass = augment_result["k_changed"] and augment_result["v_changed"]
    print(f"  K changed: {augment_result['k_changed']} (rel change: {augment_result['k_relative_change']:.4f})")
    print(f"  V changed: {augment_result['v_changed']} (rel change: {augment_result['v_relative_change']:.4f})")
    print(f"  Result: {'PASS' if augment_pass else 'FAIL'}")
    print()

    # --- V-Basis Orthonormality ---
    print("V-basis orthonormality verification...")
    ortho_pass = verify_v_basis_orthonormality()
    print(f"  Result: {'PASS' if ortho_pass else 'FAIL'}")
    print()

    # --- Performance Overhead ---
    print("Fast weights overhead benchmark (20-run avg, warmup=3):")
    perf_results = benchmark_fast_weights_overhead()
    print(f"  {'Rank':>8} {'Total ms':>12} {'ms/token':>10}")
    print(f"  {'-'*8} {'-'*12} {'-'*10}")
    for rank_key, metrics in perf_results.items():
        print(f"  {rank_key:>8} {metrics['total_ms']:>11.2f} {metrics['ms_per_token']:>9.2f}")
    print()

    # --- Save results ---
    results = {
        "baseline_identity": identity_pass,
        "augment_effect": augment_result,
        "v_basis_orthonormal": ortho_pass,
        "performance": perf_results,
    }

    output_path = RESULTS_DIR / "results.json"
    with open(output_path, "w") as f:
        json.dump(results, f, indent=2)
    print(f"Results saved to {output_path}")

    all_pass = identity_pass and augment_pass and ortho_pass
    if all_pass:
        print("\nStage 4 VERIFIED: Fast weights structure works correctly, baseline identity preserved.")
    else:
        print("\nStage 4 FAILED: Check results above.")


if __name__ == "__main__":
    main()
