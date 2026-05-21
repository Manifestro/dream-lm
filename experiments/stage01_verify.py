"""
Stage 1 verification: gradient check + forward pass benchmarks.

Run: uv run python -m experiments.stage01_verify
"""

import time

import torch
import torch.nn as nn
from torch.autograd import gradcheck

from dream_lm.core.transformer_layer import TransformerLayer


def grad_check() -> bool:
    """Verify input gradients via torch.autograd.gradcheck.

    Checks that the analytical gradient of the output w.r.t. the input x
    matches the numerical (finite-difference) Jacobian. Weight gradients
    are verified separately by test_gradient_flows in the test suite.
    """
    d_model, n_heads = 16, 2
    layer = TransformerLayer(d_model, n_heads).double()

    x = torch.randn(1, 4, d_model, dtype=torch.double, requires_grad=True)

    def fn(x_in: torch.Tensor) -> torch.Tensor:
        output, _ = layer(x_in)
        return output

    try:
        gradcheck(fn, (x,), atol=1e-4, rtol=1e-3)
        print("[✓] Gradient check passed")
        return True
    except Exception as e:
        print(f"[✗] Gradient check failed: {e}")
        return False


def benchmark_single() -> list[dict]:
    """Forward pass benchmark for a single TransformerLayer (inference baseline)."""
    configs = [
        # (batch, seq_len, d_model, n_heads)
        (1, 64, 128, 4),
        (4, 128, 256, 8),
        (2, 256, 512, 8),
    ]
    results = []
    for batch, seq_len, d_model, n_heads in configs:
        layer = TransformerLayer(d_model, n_heads)
        x = torch.randn(batch, seq_len, d_model)

        with torch.no_grad():
            for _ in range(3):
                layer(x)

        n_runs = 20
        with torch.no_grad():
            start = time.perf_counter()
            for _ in range(n_runs):
                layer(x)
            elapsed = (time.perf_counter() - start) / n_runs

        results.append({
            "batch": batch, "seq_len": seq_len,
            "d_model": d_model, "n_heads": n_heads,
            "forward_ms": elapsed * 1000,
        })
        print(f"  1L ({batch}, {seq_len}, {d_model}, {n_heads}) → {elapsed*1000:.2f} ms")

    return results


def benchmark_stack() -> list[dict]:
    """Forward pass benchmark for stacked TransformerLayers.

    Baseline for Stage 17 (multi-layer stack). Fixed config so numbers
    are directly comparable across stages.
    """
    d_model, n_heads, batch, seq_len = 256, 8, 4, 128
    results = []

    for n_layers in (1, 4, 6):
        stack = nn.ModuleList([TransformerLayer(d_model, n_heads) for _ in range(n_layers)])

        x = torch.randn(batch, seq_len, d_model)

        with torch.no_grad():
            for _ in range(3):
                out = x
                for layer in stack:
                    out, _ = layer(out)

        n_runs = 20
        with torch.no_grad():
            start = time.perf_counter()
            for _ in range(n_runs):
                out = x
                for layer in stack:
                    out, _ = layer(out)
            elapsed = (time.perf_counter() - start) / n_runs

        results.append({
            "n_layers": n_layers,
            "batch": batch, "seq_len": seq_len,
            "d_model": d_model, "n_heads": n_heads,
            "forward_ms": elapsed * 1000,
        })
        print(f"  {n_layers}L ({batch}, {seq_len}, {d_model}, {n_heads}) → {elapsed*1000:.2f} ms")

    return results


if __name__ == "__main__":
    print("=" * 50)
    print("Stage 1 Verification")
    print("=" * 50)

    print("\n--- Gradient Check ---")
    grad_check()

    print("\n--- Single Layer Benchmark (CPU, no_grad) ---")
    benchmark_single()

    print("\n--- Stack Benchmark — Stage 17 baseline (CPU, no_grad) ---")
    benchmark_stack()

    print("\n[✓] Stage 1 verification complete")
