"""
Stage 3 tests — KV-Cache for incremental inference.

Tests cover:
- KVCache dataclass (init, append, FIFO truncation)
- Output numerical identity (with vs without cache)
- Cache shape and growth
- Speedup verification
- Gradient flow (training path unchanged)
- Positional encoding offset
"""

import math
import time

import pytest
import torch
import torch.nn as nn
import torch.nn.functional as F

from dream_lm.core.kv_cache import KVCache
from dream_lm.core.model import DREAMLM
from dream_lm.core.positional_encoding import SinusoidalPositionalEncoding
from dream_lm.core.transformer_layer import TransformerLayer


# ---------------------------------------------------------------------------
# Helper: create a small model for fast testing
# ---------------------------------------------------------------------------
def _make_model(vocab_size=65, d_model=64, n_heads=4, n_layers=2, max_seq_len=256):
    return DREAMLM(
        vocab_size=vocab_size,
        d_model=d_model,
        n_heads=n_heads,
        n_layers=n_layers,
        ff_mult=4,
        max_seq_len=max_seq_len,
        activation="gelu",
        tie_embeddings=True,
    )


# ---------------------------------------------------------------------------
# KVCache dataclass tests
# ---------------------------------------------------------------------------
class TestKVCacheInit:
    def test_empty_cache_has_zero_seq_len(self):
        cache = KVCache.init(batch=1, n_heads=4, d_head=16)
        assert cache.seq_len == 0
        assert cache.is_empty  # zero tokens stored

    def test_not_empty_after_append(self):
        cache = KVCache.init(batch=1, n_heads=4, d_head=16)
        assert cache.is_empty
        cache.append(torch.randn(1, 4, 1, 16), torch.randn(1, 4, 1, 16))
        assert not cache.is_empty

    def test_cache_shapes(self):
        cache = KVCache.init(batch=2, n_heads=8, d_head=32)
        assert cache.k.shape == (2, 8, 0, 32)
        assert cache.v.shape == (2, 8, 0, 32)

    def test_device_placement(self):
        cache = KVCache.init(batch=1, n_heads=4, d_head=16, device="cpu")
        assert cache.k.device.type == "cpu"


class TestKVCacheAppend:
    def test_append_grows_seq_len(self):
        cache = KVCache.init(batch=1, n_heads=4, d_head=16)
        k_new = torch.randn(1, 4, 1, 16)
        v_new = torch.randn(1, 4, 1, 16)
        cache.append(k_new, v_new)
        assert cache.seq_len == 1

        k_new2 = torch.randn(1, 4, 1, 16)
        v_new2 = torch.randn(1, 4, 1, 16)
        cache.append(k_new2, v_new2)
        assert cache.seq_len == 2

    def test_append_first_token(self):
        """First append should set the cache directly (no concat with empty)."""
        cache = KVCache.init(batch=1, n_heads=2, d_head=8)
        k_new = torch.randn(1, 2, 1, 8)
        v_new = torch.randn(1, 2, 1, 8)
        cache.append(k_new, v_new)
        assert torch.allclose(cache.k, k_new)
        assert torch.allclose(cache.v, v_new)

    def test_fifo_truncation(self):
        cache = KVCache.init(batch=1, n_heads=2, d_head=8, max_cache_len=3)
        for _ in range(5):
            cache.append(torch.randn(1, 2, 1, 8), torch.randn(1, 2, 1, 8))
        assert cache.seq_len == 3


# ---------------------------------------------------------------------------
# Positional encoding offset tests
# ---------------------------------------------------------------------------
class TestPEOffset:
    def test_offset_zero_same_as_before(self):
        pe = SinusoidalPositionalEncoding(d_model=64, max_seq_len=128)
        x = torch.randn(1, 10, 64)
        y1 = pe(x, offset=0)
        y2 = pe(x)  # default offset=0
        assert torch.allclose(y1, y2)

    def test_offset_changes_positions(self):
        pe = SinusoidalPositionalEncoding(d_model=64, max_seq_len=128)
        x = torch.ones(1, 1, 64)
        y0 = pe(x, offset=0)
        y5 = pe(x, offset=5)
        # Same input, different PE → different output
        assert not torch.allclose(y0, y5)

    def test_offset_exceeds_max_raises(self):
        pe = SinusoidalPositionalEncoding(d_model=64, max_seq_len=32)
        x = torch.randn(1, 1, 64)
        with pytest.raises(AssertionError):
            pe(x, offset=32)  # 32 + 1 > 32


# ---------------------------------------------------------------------------
# Output identity: cached generation must match non-cached generation
# ---------------------------------------------------------------------------
class TestOutputIdentity:
    def _generate_without_cache(self, model, prompt, max_new_tokens, temperature=0.0):
        """Old-style generation: re-process full context each step."""
        tokens = list(prompt)
        for _ in range(max_new_tokens):
            context = tokens[-model.pe.max_seq_len:]
            x = torch.tensor([context], dtype=torch.long)
            with torch.no_grad():
                out = model(x)
            logits = out.logits[0, -1, :]
            if temperature == 0.0:
                next_token = logits.argmax().item()
            else:
                probs = torch.softmax(logits / temperature, dim=-1)
                next_token = torch.multinomial(probs, 1).item()
            tokens.append(next_token)
        return tokens

    def test_greedy_output_identical(self):
        """With temperature=0 (greedy), cached and non-cached must be identical."""
        torch.manual_seed(42)
        model = _make_model()
        model.eval()

        prompt = [10, 20, 30, 40, 50]

        with torch.no_grad():
            tokens_cached = model.generate(prompt, max_new_tokens=10, temperature=0.0)
            tokens_uncached = self._generate_without_cache(model, prompt, max_new_tokens=10, temperature=0.0)

        assert tokens_cached == tokens_uncached, (
            f"Cached: {tokens_cached}\nUncached: {tokens_uncached}"
        )

    def test_greedy_multiple_seeds(self):
        """Test identity across multiple random seeds."""
        for seed in [0, 1, 7, 99]:
            torch.manual_seed(seed)
            model = _make_model()
            model.eval()

            prompt = [i % 65 for i in range(8)]

            with torch.no_grad():
                tokens_cached = model.generate(prompt, max_new_tokens=5, temperature=0.0)
                tokens_uncached = self._generate_without_cache(model, prompt, max_new_tokens=5, temperature=0.0)

            assert tokens_cached == tokens_uncached, f"Seed {seed}: cached != uncached"

    def test_greedy_different_lengths(self):
        """Test identity at different context lengths."""
        torch.manual_seed(123)
        model = _make_model()
        model.eval()

        for prompt_len in [1, 5, 20, 50]:
            prompt = [i % 65 for i in range(prompt_len)]

            with torch.no_grad():
                tokens_cached = model.generate(prompt, max_new_tokens=5, temperature=0.0)
                tokens_uncached = self._generate_without_cache(model, prompt, max_new_tokens=5, temperature=0.0)

            assert tokens_cached == tokens_uncached, f"Prompt len {prompt_len}: cached != uncached"


# ---------------------------------------------------------------------------
# Cache shape and growth tests
# ---------------------------------------------------------------------------
class TestCacheShape:
    def test_cache_grows_by_one_per_step(self):
        """Each generation step should increase cache seq_len by 1."""
        torch.manual_seed(0)
        model = _make_model()
        model.eval()

        prompt = [10, 20, 30]
        context = prompt[-model.pe.max_seq_len:]
        x = torch.tensor([context], dtype=torch.long)

        h = model.embedding(x)
        h = model.pe(h)

        kv_caches = [
            KVCache.init(batch=1, n_heads=model.layers[0].attn.n_heads, d_head=model.layers[0].attn.d_head)
            for _ in range(len(model.layers))
        ]

        with torch.no_grad():
            initial_len = len(context)
            _, kv_caches, _, _ = model.forward_with_cache(h, kv_caches)
            assert kv_caches[0].seq_len == initial_len

            # Generate 3 tokens
            for step in range(3):
                next_id = [torch.randint(0, 65, (1,)).item()]
                x_next = torch.tensor([next_id], dtype=torch.long)
                h_next = model.embedding(x_next)
                h_next = model.pe(h_next, offset=initial_len + step)
                _, kv_caches, _, _ = model.forward_with_cache(h_next, kv_caches)
                assert kv_caches[0].seq_len == initial_len + step + 1


# ---------------------------------------------------------------------------
# Speedup test
# ---------------------------------------------------------------------------
class TestSpeedup:
    def test_cached_faster_than_uncached(self):
        """Cached generation should be faster for sequences longer than prompt."""
        torch.manual_seed(0)
        model = _make_model(d_model=128, n_heads=4, n_layers=4)
        model.eval()

        prompt = [i % 65 for i in range(32)]
        n_tokens = 20

        # Cached
        torch.manual_seed(0)
        t0 = time.perf_counter()
        with torch.no_grad():
            model.generate(prompt, max_new_tokens=n_tokens, temperature=0.0)
        cached_time = time.perf_counter() - t0

        # Uncached (manual re-processing)
        torch.manual_seed(0)
        t0 = time.perf_counter()
        tokens = list(prompt)
        for _ in range(n_tokens):
            context = tokens[-model.pe.max_seq_len:]
            x = torch.tensor([context], dtype=torch.long)
            with torch.no_grad():
                out = model(x)
            next_token = out.logits[0, -1, :].argmax().item()
            tokens.append(next_token)
        uncached_time = time.perf_counter() - t0

        # Cached should be at least 1.5x faster
        speedup = uncached_time / cached_time
        assert speedup > 1.5, f"Speedup {speedup:.2f}x < 1.5x (cached={cached_time:.4f}s, uncached={uncached_time:.4f}s)"


# ---------------------------------------------------------------------------
# Gradient flow: training path must not be affected by KV-Cache additions
# ---------------------------------------------------------------------------
class TestGradientFlow:
    def test_backward_without_cache(self):
        """Training forward pass (no cache) should still produce gradients."""
        torch.manual_seed(0)
        model = _make_model()
        model.train()

        x = torch.randint(0, 65, (2, 16))
        out = model(x)  # no cache path
        loss = F.cross_entropy(out.logits.view(-1, 65), torch.randint(0, 65, (2, 16)).view(-1))
        loss.backward()

        # All parameters should have gradients
        for name, param in model.named_parameters():
            assert param.grad is not None, f"No gradient for {name}"
            assert not torch.isnan(param.grad).any(), f"NaN gradient for {name}"

    def test_forward_with_cache_does_not_break_training(self):
        """Calling forward_with_cache should not affect subsequent training forward."""
        torch.manual_seed(0)
        model = _make_model()
        model.eval()

        # First, do a cached forward pass
        prompt = [10, 20, 30]
        with torch.no_grad():
            model.generate(prompt, max_new_tokens=3, temperature=0.0)

        # Now switch to training mode and verify backward works
        model.train()
        x = torch.randint(0, 65, (2, 16))
        out = model(x)
        loss = F.cross_entropy(out.logits.view(-1, 65), torch.randint(0, 65, (2, 16)).view(-1))
        loss.backward()

        for name, param in model.named_parameters():
            assert param.grad is not None, f"No gradient for {name} after cached generation"
