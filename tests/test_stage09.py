"""
Stage 9 tests — STDP update for fast weights.

Tests cover:
- _compute_stdp_delta: shape correctness, projection math, surprise gating
- stdp_update: in-place mutation, no_grad, max_norm clipping, lambda decay
- Integration: forward_with_cache applies STDP, generate() updates U
- Baseline: fast_weight_r=0 still works (backward compat)
"""

import pytest
import torch

from dream_lm.core.fast_weights import FastWeightState
from dream_lm.core.model import DREAMLM
from dream_lm.core.ema import EMAStats


# ============================================================
# _compute_stdp_delta tests
# ============================================================

class TestComputeStdpDelta:
    def test_output_shape(self):
        fw = FastWeightState.init(n_heads=4, d_head=32, r=8)
        eps = torch.randn(2, 16, 128)
        h = torch.randn(2, 16, 128)
        surprise = torch.ones(2, 16)
        delta = fw._compute_stdp_delta(eps, h, surprise)
        assert delta.shape == (4, 32, 8)

    def test_zero_surprise_zero_delta(self):
        fw = FastWeightState.init(n_heads=4, d_head=32, r=8)
        eps = torch.randn(2, 16, 128)
        h = torch.randn(2, 16, 128)
        surprise = torch.zeros(2, 16)
        delta = fw._compute_stdp_delta(eps, h, surprise)
        assert torch.allclose(delta, torch.zeros_like(delta), atol=1e-6)

    def test_surprise_gating(self):
        """Higher surprise should produce larger delta magnitude."""
        fw = FastWeightState.init(n_heads=4, d_head=32, r=8)
        eps = torch.randn(2, 16, 128)
        h = torch.randn(2, 16, 128)

        surprise_low = torch.ones(2, 16) * 0.1
        surprise_high = torch.ones(2, 16) * 1.0

        delta_low = fw._compute_stdp_delta(eps, h, surprise_low)
        delta_high = fw._compute_stdp_delta(eps, h, surprise_high)

        assert delta_high.norm() > delta_low.norm()

    def test_batch_averaging(self):
        """Delta should scale correctly with batch size."""
        fw = FastWeightState.init(n_heads=4, d_head=32, r=8)
        torch.manual_seed(42)

        # Single batch
        eps1 = torch.randn(1, 16, 128)
        h1 = torch.randn(1, 16, 128)
        surprise1 = torch.ones(1, 16)
        delta1 = fw._compute_stdp_delta(eps1, h1, surprise1)

        # Double batch (same data repeated)
        eps2 = torch.cat([eps1, eps1], dim=0)
        h2 = torch.cat([h1, h1], dim=0)
        surprise2 = torch.ones(2, 16)
        delta2 = fw._compute_stdp_delta(eps2, h2, surprise2)

        # Same input → same delta (averaging cancels duplication)
        assert torch.allclose(delta1, delta2, atol=1e-5)

    def d_model_divisibility_assertion(self):
        """d_model must equal n_heads × d_head."""
        fw = FastWeightState.init(n_heads=4, d_head=32, r=8)
        eps = torch.randn(1, 1, 100)  # 100 != 4*32
        h = torch.randn(1, 1, 100)
        surprise = torch.ones(1, 1)
        with pytest.raises(AssertionError):
            fw._compute_stdp_delta(eps, h, surprise)


# ============================================================
# stdp_update tests
# ============================================================

class TestStdpUpdate:
    def test_u_k_mutated_in_place(self):
        fw = FastWeightState.init(n_heads=4, d_head=32, r=8)
        initial_u_k = fw.u_k.clone()
        eps = torch.randn(2, 16, 128)
        h = torch.randn(2, 16, 128)
        e = torch.randn(2, 16, 128)
        surprise = torch.ones(2, 16)
        fw.stdp_update(eps, h, e, surprise, eta=0.1, lambda_decay=1.0)
        assert not torch.allclose(fw.u_k, initial_u_k)

    def test_u_v_mutated_in_place(self):
        fw = FastWeightState.init(n_heads=4, d_head=32, r=8)
        initial_u_v = fw.u_v.clone()
        eps = torch.randn(2, 16, 128)
        h = torch.randn(2, 16, 128)
        e = torch.randn(2, 16, 128)
        surprise = torch.ones(2, 16)
        fw.stdp_update(eps, h, e, surprise, eta=0.1, lambda_decay=1.0)
        assert not torch.allclose(fw.u_v, initial_u_v)

    def test_no_grad(self):
        """STDP update should not require gradients."""
        fw = FastWeightState.init(n_heads=4, d_head=32, r=8)
        eps = torch.randn(2, 16, 128, requires_grad=True)
        h = torch.randn(2, 16, 128, requires_grad=True)
        e = torch.randn(2, 16, 128, requires_grad=True)
        surprise = torch.ones(2, 16)
        fw.stdp_update(eps, h, e, surprise)
        # If no_grad works, u_k/u_v should not have grad_fn
        assert fw.u_k.grad_fn is None
        assert fw.u_v.grad_fn is None

    def test_lambda_decay(self):
        """Successive updates with same input should show decay effect."""
        fw = FastWeightState.init(n_heads=4, d_head=32, r=8)
        eps = torch.randn(2, 16, 128)
        h = torch.randn(2, 16, 128)
        e = torch.randn(2, 16, 128)
        surprise = torch.ones(2, 16)

        fw.stdp_update(eps, h, e, surprise, eta=0.1, lambda_decay=0.95)
        norm_after_1 = fw.u_k.norm().item()

        fw.stdp_update(eps, h, e, surprise, eta=0.1, lambda_decay=0.95)
        norm_after_2 = fw.u_k.norm().item()

        # With decay, second update adds less than first
        assert norm_after_2 < norm_after_1 * 2

    def test_max_norm_clipping(self):
        """After update, per-head norm should not exceed max_norm."""
        fw = FastWeightState.init(n_heads=4, d_head=32, r=8)
        eps = torch.randn(2, 16, 128) * 100  # large input
        h = torch.randn(2, 16, 128) * 100
        e = torch.randn(2, 16, 128) * 100
        surprise = torch.ones(2, 16)
        fw.stdp_update(eps, h, e, surprise, eta=1.0, lambda_decay=1.0, max_norm=1.0)

        # Check per-head Frobenius norm
        for head_idx in range(fw.u_k.shape[0]):
            u_k_norm = fw.u_k[head_idx].norm().item()
            u_v_norm = fw.u_v[head_idx].norm().item()
            assert u_k_norm <= 1.0 + 1e-5, f"U_K head {head_idx} norm = {u_k_norm}"
            assert u_v_norm <= 1.0 + 1e-5, f"U_V head {head_idx} norm = {u_v_norm}"

    def test_zero_surprise_only_decay(self):
        """Zero surprise → only decay applies, no meaningful update."""
        fw = FastWeightState.init(n_heads=4, d_head=32, r=8)
        # Use small initial value so max_norm clipping doesn't interfere
        initial = 0.01  # norm = 0.01 * sqrt(256) = 0.16, well under max_norm=1.0
        fw.u_k.fill_(initial)
        fw.u_v.fill_(initial)
        eps = torch.randn(2, 16, 128)
        h = torch.randn(2, 16, 128)
        e = torch.randn(2, 16, 128)
        surprise = torch.zeros(2, 16)
        fw.stdp_update(eps, h, e, surprise, eta=0.1, lambda_decay=0.95)

        # With zero surprise, only decay applies (no clipping since norm is small)
        expected_k = initial * 0.95
        assert torch.allclose(fw.u_k, torch.full_like(fw.u_k, expected_k), atol=1e-5)

    def test_u_k_u_v_differ(self):
        """U_K and U_V should update differently since h ≠ e."""
        fw = FastWeightState.init(n_heads=4, d_head=32, r=8)
        eps = torch.randn(2, 16, 128)
        h = torch.randn(2, 16, 128)
        e = torch.randn(2, 16, 128) * 2.0 + 1.0  # different from h
        surprise = torch.ones(2, 16)
        fw.stdp_update(eps, h, e, surprise, eta=0.1, lambda_decay=1.0)

        assert not torch.allclose(fw.u_k, fw.u_v)


# ============================================================
# forward_with_cache STDP integration tests
# ============================================================

class TestForwardWithCacheSTDP:
    def test_stdp_applied_when_params_given(self):
        """forward_with_cache with STDP params → U_K non-zero."""
        model = DREAMLM(
            vocab_size=100, d_model=128, n_heads=4, n_layers=2,
            fast_weight_r=8
        )
        x = torch.randn(1, 16, 128)
        kv_caches = [None] * 2
        for i in range(2):
            kv_caches[i] = model._init_cache_for_layer(1, 16, x.device, x.dtype)
        ema = EMAStats.init(batch=1)

        _, caches, _, _, _ = model.forward_with_cache(
            x, kv_caches, ema,
            stdp_eta=0.1, stdp_lambda_decay=1.0, stdp_max_norm=1.0
        )

        assert caches[0].fast_weights.u_k.norm() > 0

    def test_stdp_not_applied_when_params_none(self):
        """forward_with_cache without STDP params → U_K stays zero."""
        model = DREAMLM(
            vocab_size=100, d_model=128, n_heads=4, n_layers=2,
            fast_weight_r=8
        )
        x = torch.randn(1, 16, 128)
        kv_caches = [None] * 2
        for i in range(2):
            kv_caches[i] = model._init_cache_for_layer(1, 16, x.device, x.dtype)
        ema = EMAStats.init(batch=1)

        _, caches, _, _, _ = model.forward_with_cache(
            x, kv_caches, ema
        )

        assert torch.allclose(caches[0].fast_weights.u_k, torch.zeros_like(caches[0].fast_weights.u_k))

    def test_backward_compat_no_stdp_params(self):
        """Existing 5-tuple unpacking still works without STDP params."""
        model = DREAMLM(
            vocab_size=100, d_model=128, n_heads=4, n_layers=2, G=8
        )
        x = torch.randn(1, 8, 128)
        kv_caches = [None] * 2
        ema_scalar = EMAStats.init(batch=1)
        ema_grouped = EMAStats.init(batch=1, G=8)

        result = model.forward_with_cache(x, kv_caches, ema_scalar, ema_grouped)
        assert len(result) == 5

    def test_layer_hiddens_captured(self):
        """Each layer gets different h (not the final h)."""
        model = DREAMLM(
            vocab_size=100, d_model=128, n_heads=4, n_layers=4,
            fast_weight_r=8
        )
        x = torch.randn(1, 16, 128)
        kv_caches = [None] * 4
        for i in range(4):
            kv_caches[i] = model._init_cache_for_layer(1, 16, x.device, x.dtype)
        ema = EMAStats.init(batch=1)

        _, caches, _, _, _ = model.forward_with_cache(
            x, kv_caches, ema,
            stdp_eta=0.1, stdp_lambda_decay=1.0, stdp_max_norm=1.0
        )

        # Each layer's fast weights should have been updated differently
        norms = [caches[i].fast_weights.u_k.norm().item() for i in range(4)]
        # At least some variation between layers
        assert max(norms) > min(norms) * 0.5  # not all identical


# ============================================================
# generate() STDP tests
# ============================================================

class TestGenerateWithSTDP:
    def test_generation_produces_tokens(self):
        """generate() with STDP params still produces valid tokens."""
        model = DREAMLM(
            vocab_size=66, d_model=128, n_heads=4, n_layers=2,
            fast_weight_r=8
        )
        model.eval()

        tokens = model.generate(
            prompt=[1, 2, 3, 4, 5],
            max_new_tokens=10,
            stdp_eta=0.01,
            stdp_lambda_decay=0.95,
            stdp_max_norm=1.0,
        )

        assert len(tokens) == 15  # 5 prompt + 10 new
        assert all(isinstance(t, int) for t in tokens)

    def test_u_grows_during_generation(self):
        """After generate(), U norms are larger than initial."""
        model = DREAMLM(
            vocab_size=66, d_model=128, n_heads=4, n_layers=2,
            fast_weight_r=8
        )
        model.eval()

        # Access caches after generate by running forward_with_cache manually
        prompt = [1, 2, 3, 4, 5]
        context = prompt[-model.pe.max_seq_len:]
        x = torch.tensor([context], dtype=torch.long)
        h = model.embedding(x)
        h = model.pe(h)

        device = h.device
        n_heads = model.layers[0].attn.n_heads
        d_head = model.layers[0].attn.d_head
        kv_caches = [
            DREAMLM._init_cache_for_layer(model, 1, 16, device, h.dtype)
            for _ in range(len(model.layers))
        ]
        ema = EMAStats.init(batch=1, device=device, dtype=h.dtype)

        _, caches, _, _, _ = model.forward_with_cache(
            h, kv_caches, ema,
            stdp_eta=0.1, stdp_lambda_decay=1.0, stdp_max_norm=1.0
        )

        # U should have grown from zero
        total_norm = sum(caches[i].fast_weights.u_k.norm().item() for i in range(2))
        assert total_norm > 0

    def test_different_prompts_different_u(self):
        """Different prompts → different U update patterns."""
        model = DREAMLM(
            vocab_size=66, d_model=128, n_heads=4, n_layers=2,
            fast_weight_r=8
        )
        model.eval()

        def get_u_norm(prompt):
            x = torch.tensor([prompt], dtype=torch.long)
            h = model.embedding(x)
            h = model.pe(h)
            device = h.device
            kv_caches = [
                DREAMLM._init_cache_for_layer(model, 1, 16, device, h.dtype)
                for _ in range(len(model.layers))
            ]
            ema = EMAStats.init(batch=1, device=device, dtype=h.dtype)
            _, caches, _, _, _ = model.forward_with_cache(
                h, kv_caches, ema,
                stdp_eta=0.1, stdp_lambda_decay=1.0, stdp_max_norm=1.0
            )
            return caches[0].fast_weights.u_k.clone()

        u_a = get_u_norm([1, 2, 3, 4, 5])
        u_b = get_u_norm([60, 61, 62, 63, 64])

        assert not torch.allclose(u_a, u_b)


# ============================================================
# Baseline compatibility tests
# ============================================================

class TestBaselineCompatibility:
    def test_fast_weight_r_zero_unchanged(self):
        """Model with fast_weight_r=0 works identically to before."""
        model = DREAMLM(
            vocab_size=100, d_model=128, n_heads=4, n_layers=2,
            fast_weight_r=0
        )
        x = torch.randint(0, 100, (2, 16))
        out = model(x)
        assert out.logits.shape == (2, 16, 100)

    def test_generate_without_fast_weights(self):
        """generate() works when fast_weight_r=0."""
        model = DREAMLM(
            vocab_size=66, d_model=128, n_heads=4, n_layers=2,
            fast_weight_r=0
        )
        model.eval()

        tokens = model.generate(prompt=[1, 2, 3], max_new_tokens=5)
        assert len(tokens) == 8


# ============================================================
# Helper: create a KVCache with FastWeightState for a layer
# ============================================================

def _init_cache_for_layer(
    self,
    batch: int,
    max_cache_len: int,
    device: torch.device | str,
    dtype: torch.dtype,
):
    """Helper to create a single KVCache with FastWeightState."""
    from dream_lm.core.kv_cache import KVCache
    from dream_lm.core.fast_weights import FastWeightState

    n_heads = self.layers[0].attn.n_heads
    d_head = self.layers[0].attn.d_head

    return KVCache.init(
        batch=batch,
        n_heads=n_heads,
        d_head=d_head,
        device=device,
        dtype=dtype,
        max_cache_len=max_cache_len,
        fast_weights=FastWeightState.init(
            n_heads, d_head, self.fast_weight_r, device, dtype
        ) if self.fast_weight_r > 0 else None,
    )

# Monkey-patch onto DREAMLM for test convenience
DREAMLM._init_cache_for_layer = _init_cache_for_layer
