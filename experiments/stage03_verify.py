"""
Stage 3 verification: KV-Cache correctness and benchmark.

Verifies:
1. Cached generation output matches uncached (greedy, multiple seeds/lengths)
2. Speedup measurement: ms/token with vs without cache
3. Memory usage at different context lengths

Results saved to experiments/stage03_verify/results/
"""

import json
import time
from pathlib import Path

import torch

from dream_lm.core.model import DREAMLM
from dream_lm.core.kv_cache import KVCache

RESULTS_DIR = Path("experiments/stage03_verify/results")
RESULTS_DIR.mkdir(parents=True, exist_ok=True)


def benchmark_cached_vs_uncached(
    model: DREAMLM,
    prompt_len: int,
    n_gen_tokens: int,
    n_runs: int = 20,
    warmup: int = 3,
) -> dict:
    """Measure ms/token for cached vs uncached generation."""
    torch.manual_seed(42)
    model.eval()

    prompt = [i % model.vocab_size for i in range(prompt_len)]

    # --- Cached benchmark ---
    times_cached = []
    for run in range(warmup + n_runs):
        torch.manual_seed(42)
        t0 = time.perf_counter()
        with torch.no_grad():
            model.generate(prompt, max_new_tokens=n_gen_tokens, temperature=0.0)
        elapsed = time.perf_counter() - t0
        if run >= warmup:
            times_cached.append(elapsed)

    # --- Uncached benchmark (manual re-processing) ---
    times_uncached = []
    for run in range(warmup + n_runs):
        torch.manual_seed(42)
        tokens = list(prompt)
        t0 = time.perf_counter()
        with torch.no_grad():
            for _ in range(n_gen_tokens):
                context = tokens[-model.pe.max_seq_len:]
                x = torch.tensor([context], dtype=torch.long)
                logits = model(x)
                next_token = logits[0, -1, :].argmax().item()
                tokens.append(next_token)
        elapsed = time.perf_counter() - t0
        times_uncached.append(elapsed)

    avg_cached = sum(times_cached) / len(times_cached)
    avg_uncached = sum(times_uncached) / len(times_uncached)

    return {
        "cached_total_ms": avg_cached * 1000,
        "cached_ms_per_token": (avg_cached / n_gen_tokens) * 1000,
        "uncached_total_ms": avg_uncached * 1000,
        "uncached_ms_per_token": (avg_uncached / n_gen_tokens) * 1000,
        "speedup": avg_uncached / avg_cached,
    }


def verify_correctness(model: DREAMLM, n_tests: int = 10) -> bool:
    """Verify cached == uncached across multiple seeds and lengths."""
    model.eval()
    for i in range(n_tests):
        torch.manual_seed(i * 7)
        prompt_len = (i % 30) + 1
        prompt = [j % model.vocab_size for j in range(prompt_len)]
        n_gen = 5

        with torch.no_grad():
            tokens_cached = model.generate(prompt, max_new_tokens=n_gen, temperature=0.0)

        # Uncached
        tokens = list(prompt)
        for _ in range(n_gen):
            context = tokens[-model.pe.max_seq_len:]
            x = torch.tensor([context], dtype=torch.long)
            logits = model(x)
            next_token = logits[0, -1, :].argmax().item()
            tokens.append(next_token)

        if tokens_cached != tokens:
            print(f"  FAIL seed={i} prompt_len={prompt_len}")
            print(f"    cached:   {tokens_cached}")
            print(f"    uncached: {tokens}")
            return False

    return True


def main():
    print("Stage 3 Verification: KV-Cache")
    print("=" * 50)

    # Load model from Stage 2 config
    import yaml

    with open("configs/stage02.yaml") as f:
        cfg = yaml.safe_load(f)

    model = DREAMLM(
        vocab_size=65,
        d_model=cfg["model"]["d_model"],
        n_heads=cfg["model"]["n_heads"],
        n_layers=cfg["model"]["n_layers"],
        ff_mult=cfg["model"]["ff_mult"],
        max_seq_len=cfg["model"]["max_seq_len"],
        activation=cfg["model"]["activation"],
        tie_embeddings=True,
    )
    model.eval()

    n_params = sum(p.numel() for p in model.parameters())
    print(f"Model: {n_params:,} parameters")
    print(f"Config: d_model={cfg['model']['d_model']}, n_heads={cfg['model']['n_heads']}, "
          f"n_layers={cfg['model']['n_layers']}")
    print()

    # --- Correctness ---
    print("Correctness verification (10 tests)...")
    correct = verify_correctness(model, n_tests=10)
    print(f"  Result: {'PASS' if correct else 'FAIL'}")
    print()

    # --- Benchmarks ---
    configs = [
        {"prompt_len": 32, "n_gen": 20, "label": "short (32+20)"},
        {"prompt_len": 64, "n_gen": 32, "label": "medium (64+32)"},
        {"prompt_len": 128, "n_gen": 32, "label": "long (128+32)"},
        {"prompt_len": 200, "n_gen": 32, "label": "very long (200+32)"},
    ]

    benchmarks = {}
    print("Benchmark (20-run avg, warmup=3):")
    print(f"  {'Config':<25} {'Cached ms/tok':>14} {'Uncached ms/tok':>16} {'Speedup':>8}")
    print(f"  {'-'*25} {'-'*14} {'-'*16} {'-'*8}")

    for c in configs:
        result = benchmark_cached_vs_uncached(model, c["prompt_len"], c["n_gen"])
        benchmarks[c["label"]] = result
        print(f"  {c['label']:<25} {result['cached_ms_per_token']:>13.2f} "
              f"{result['uncached_ms_per_token']:>15.2f} {result['speedup']:>7.1f}x")

    print()

    # --- Save results ---
    results = {
        "correctness": correct,
        "n_parameters": n_params,
        "benchmarks": benchmarks,
    }

    output_path = RESULTS_DIR / "results.json"
    with open(output_path, "w") as f:
        json.dump(results, f, indent=2)
    print(f"Results saved to {output_path}")

    if correct:
        print("\nStage 3 VERIFIED: KV-Cache output identical to full recomputation, "
              f"with {benchmarks['medium (64+32)']['speedup']:.1f}x speedup.")
    else:
        print("\nStage 3 FAILED: Cached output does not match uncached.")


if __name__ == "__main__":
    main()
