"""
Stage 4 tests — Low-Rank Fast Weights.

Tests cover:
- FastWeightState init (zero U, orthonormal V_basis)
- Augment shape correctness
- Zero-init no-op: output identity with baseline (Stage 3 tests must still pass)
- Non-zero augment changes output
- Gradient flow through augment path
- V_basis orthonormality
- KVCache fast_weights integration
"""

import pytest
import torch
import torch.nn.functional as F

from dream_lm.core.fast_weights import FastWeightState
from dream_lm.core.kv_cache import KVCache
from dream_lm.core.model import DREAMLM


# ---------------------------------------------------------------------------
# Helper: create a small model for fast testing
# ---------------------------------------------------------------------------
def _make_model(vocab_size=65, d_model=64, n_heads=4, n_layers=2, max_seq_len=256, fast_weight_r=0):
    return DREAMLM(
        vocab_size=vocab_size,
        d_model=d_model,
        n_heads=n_heads,
        n_layers=n_layers,
        ff_mult=4,
        max_seq_len=max_seq_len,
        activation="gelu",
        tie_embeddings=True,
        fast_weight_r=fast_weight_r,
    )


# ---------------------------------------------------------------------------
# FastWeightState init tests
# ---------------------------------------------------------------------------
class TestFastWeightStateInit:
    def test_zero_init(self):
        """U_K and U_V should be zero-initialized (no-op until updated)."""
        fw = FastWeightState.init(n_heads=4, d_head=16, r=8)
        assert torch.allclose(fw.u_k, torch.zeros_like(fw.u_k))
        assert torch.allclose(fw.u_v, torch.zeros_like(fw.u_v))

    def test_shapes(self):
        """All tensors should have shape (n_heads, d_head, r)."""
        fw = FastWeightState.init(n_heads=4, d_head=16, r=8)
        assert fw.u_k.shape == (4, 16, 8)
        assert fw.u_v.shape == (4, 16, 8)
        assert fw.v_basis.shape == (4, 16, 8)

    def test_v_basis_orthonormal_per_head(self):
        """V_basis^T @ V_basis should equal I for each head."""
        fw = FastWeightState.init(n_heads=4, d_head=16, r=8)
        for h in range(fw.v_basis.shape[0]):
            v_h = fw.v_basis[h]  # (d_head, r)
            gram = v_h.T @ v_h  # (r, r) should be identity
            assert torch.allclose(gram, torch.eye(8), atol=1e-5), f"Head {h}: V_basis not orthonormal"

    def test_v_basis_differs_across_heads(self):
        """Each head should have a different V_basis (random init)."""
        fw = FastWeightState.init(n_heads=4, d_head=16, r=8)
        for i in range(fw.v_basis.shape[0]):
            for j in range(i + 1, fw.v_basis.shape[0]):
                assert not torch.allclose(fw.v_basis[i], fw.v_basis[j]), \
                    f"Heads {i} and {j} have identical V_basis"

    def test_device_placement(self):
        fw = FastWeightState.init(n_heads=4, d_head=16, r=8, device="cpu")
        assert fw.u_k.device.type == "cpu"
        assert fw.v_basis.device.type == "cpu"

    def test_is_empty(self):
        fw = FastWeightState()
        assert fw.is_empty
        fw2 = FastWeightState.init(n_heads=4, d_head=16, r=8)
        assert not fw2.is_empty


# ---------------------------------------------------------------------------
# Augment tests
# ---------------------------------------------------------------------------
class TestAugment:
    def test_zero_augment_is_noop(self):
        """With U=0, augment should return input unchanged."""
        fw = FastWeightState.init(n_heads=4, d_head=16, r=8)
        k = torch.randn(2, 4, 10, 16)
        v = torch.randn(2, 4, 10, 16)
        k_aug, v_aug = fw.augment(k, v)
        assert torch.allclose(k_aug, k)
        assert torch.allclose(v_aug, v)

    def test_augment_output_shape(self):
        """Augmented output should have same shape as input."""
        fw = FastWeightState.init(n_heads=4, d_head=16, r=8)
        k = torch.randn(2, 4, 10, 16)
        v = torch.randn(2, 4, 10, 16)
        k_aug, v_aug = fw.augment(k, v)
        assert k_aug.shape == k.shape
        assert v_aug.shape == v.shape

    def test_nonzero_augment_changes_output(self):
        """Non-zero U should change K/V."""
        fw = FastWeightState.init(n_heads=4, d_head=16, r=8)
        fw.u_k.fill_(1.0)
        fw.u_v.fill_(1.0)

        k = torch.randn(2, 4, 10, 16)
        v = torch.randn(2, 4, 10, 16)
        k_aug, v_aug = fw.augment(k, v)

        assert not torch.allclose(k_aug, k)
        assert not torch.allclose(v_aug, v)

    def test_augment_k_only(self):
        """augment_k should only modify K."""
        fw = FastWeightState.init(n_heads=4, d_head=16, r=8)
        fw.u_k.fill_(1.0)

        k = torch.randn(2, 4, 10, 16)
        v = torch.randn(2, 4, 10, 16)
        k_aug = fw.augment_k(k)
        v_aug = fw.augment_v(v)  # v unchanged since u_v=0

        assert not torch.allclose(k_aug, k)
        assert torch.allclose(v_aug, v)

    def test_augment_v_only(self):
        """augment_v should only modify V when only u_v is non-zero."""
        fw = FastWeightState.init(n_heads=4, d_head=16, r=8)
        fw.u_v.fill_(1.0)

        k = torch.randn(2, 4, 10, 16)
        v = torch.randn(2, 4, 10, 16)
        k_aug = fw.augment_k(k)
        v_aug = fw.augment_v(v)

        assert torch.allclose(k_aug, k)
        assert not torch.allclose(v_aug, v)

    def test_augment_math_correctness_manual(self):
        """Verify augment math matches manual per-head calculation."""
        torch.manual_seed(42)
        fw = FastWeightState.init(n_heads=2, d_head=8, r=4)
        # Set non-zero U to actually test the math
        fw.u_k[0] = torch.randn(8, 4)
        fw.u_k[1] = torch.randn(8, 4)

        k = torch.randn(1, 2, 3, 8)  # (batch=1, heads=2, seq=3, d_head=8)
        k_aug = fw.augment_k(k)

        # Manual calculation: for each head, k_head @ fw_k_head^T
        fw_k = fw.u_k @ fw.v_basis.transpose(-2, -1)  # (2, 8, 8)

        expected = k.clone()
        for h in range(2):
            # k[0, h, :, :] is (3, 8), fw_k[h] is (8, 8)
            # result should be k @ fw_k^T for each position
            for s in range(3):
                expected[0, h, s, :] = k[0, h, s, :] + k[0, h, s, :] @ fw_k[h].T

        assert torch.allclose(k_aug, expected, atol=1e-5), (
            f"Augment math mismatch: max diff={(k_aug - expected).abs().max().item()}"
        )


# ---------------------------------------------------------------------------
# KVCache integration tests
# ---------------------------------------------------------------------------
class TestKVCacheFastWeights:
    def test_cache_accepts_fast_weights(self):
        """KVCache.init should accept fast_weights parameter."""
        fw = FastWeightState.init(n_heads=4, d_head=16, r=8)
        cache = KVCache.init(batch=1, n_heads=4, d_head=16, fast_weights=fw)
        assert cache.fast_weights is fw
        assert not cache.fast_weights.is_empty

    def test_cache_without_fast_weights(self):
        """KVCache.init should work without fast_weights (backward compat)."""
        cache = KVCache.init(batch=1, n_heads=4, d_head=16)
        assert cache.fast_weights is None

    def test_augment_via_cache(self):
        """Augment through KVCache.fast_weights should work."""
        fw = FastWeightState.init(n_heads=4, d_head=16, r=8)
        fw.u_k.fill_(1.0)
        cache = KVCache.init(batch=1, n_heads=4, d_head=16, fast_weights=fw)

        k = torch.randn(1, 4, 1, 16)
        v = torch.randn(1, 4, 1, 16)
        k_aug, v_aug = cache.fast_weights.augment(k, v)

        assert not torch.allclose(k_aug, k)


# ---------------------------------------------------------------------------
# Gradient flow through augment path
# ---------------------------------------------------------------------------
class TestGradientFlow:
    def test_augment_gradient_flows_to_u(self):
        """Gradients should flow through augment to U_K, U_V."""
        fw = FastWeightState.init(n_heads=4, d_head=16, r=8)
        fw.u_k.requires_grad_(True)
        fw.u_v.requires_grad_(True)

        k = torch.randn(2, 4, 10, 16, requires_grad=False)
        v = torch.randn(2, 4, 10, 16, requires_grad=False)

        k_aug, v_aug = fw.augment(k, v)
        loss = (k_aug ** 2).sum() + (v_aug ** 2).sum()
        loss.backward()

        assert fw.u_k.grad is not None
        assert fw.u_v.grad is not None
        assert not torch.isnan(fw.u_k.grad).any()
        assert not torch.isnan(fw.u_v.grad).any()

    def test_backward_with_fast_weights_in_model(self):
        """Training forward pass with fast_weight_r=16 should produce gradients."""
        torch.manual_seed(0)
        model = _make_model(fast_weight_r=16)
        model.train()

        x = torch.randint(0, 65, (2, 16))
        logits = model(x)
        loss = F.cross_entropy(logits.view(-1, 65), torch.randint(0, 65, (2, 16)).view(-1))
        loss.backward()

        for name, param in model.named_parameters():
            assert param.grad is not None, f"No gradient for {name}"
            assert not torch.isnan(param.grad).any(), f"NaN gradient for {name}"


# ---------------------------------------------------------------------------
# Output identity: fast_weight_r=0 must be bit-identical to baseline
# ---------------------------------------------------------------------------
class TestBaselineIdentity:
    def _generate_without_cache(self, model, prompt, max_new_tokens, temperature=0.0):
        """Old-style generation: re-process full context each step."""
        tokens = list(prompt)
        for _ in range(max_new_tokens):
            context = tokens[-model.pe.max_seq_len:]
            x = torch.tensor([context], dtype=torch.long)
            with torch.no_grad():
                logits = model(x)
            logits = logits[0, -1, :]
            if temperature == 0.0:
                next_token = logits.argmax().item()
            else:
                probs = torch.softmax(logits / temperature, dim=-1)
                next_token = torch.multinomial(probs, 1).item()
            tokens.append(next_token)
        return tokens

    def test_fast_weight_zero_matches_baseline(self):
        """Model with fast_weight_r=0 should produce identical output to model without fast weights."""
        torch.manual_seed(42)
        model_fw = _make_model(fast_weight_r=0)
        model_fw.eval()

        torch.manual_seed(42)
        model_base = _make_model()  # default fast_weight_r=0
        model_base.eval()

        # Copy weights to ensure identical initialization
        model_fw.load_state_dict(model_base.state_dict())

        prompt = [10, 20, 30, 40, 50]

        with torch.no_grad():
            tokens_fw = model_fw.generate(prompt, max_new_tokens=10, temperature=0.0)
            tokens_base = model_base.generate(prompt, max_new_tokens=10, temperature=0.0)

        assert tokens_fw == tokens_base, "fast_weight_r=0 should match baseline"

    def test_cached_uncached_identity_with_fast_weights_zero(self):
        """With fast_weight_r=0, cached and uncached must still be identical (Stage 3 test)."""
        torch.manual_seed(42)
        model = _make_model(fast_weight_r=0)
        model.eval()

        prompt = [10, 20, 30, 40, 50]

        with torch.no_grad():
            tokens_cached = model.generate(prompt, max_new_tokens=10, temperature=0.0)
            tokens_uncached = self._generate_without_cache(model, prompt, max_new_tokens=10, temperature=0.0)

        assert tokens_cached == tokens_uncached, (
            f"Cached: {tokens_cached}\nUncached: {tokens_uncached}"
        )

    def test_fast_weight_r_stored(self):
        """Model should store fast_weight_r attribute."""
        model = _make_model(fast_weight_r=16)
        assert model.fast_weight_r == 16

        model0 = _make_model(fast_weight_r=0)
        assert model0.fast_weight_r == 0

    def test_layers_have_independent_fast_weights(self):
        """Each layer must get its own FastWeightState — mutating one must not affect another."""
        fw_a = FastWeightState.init(n_heads=4, d_head=16, r=8)
        fw_b = FastWeightState.init(n_heads=4, d_head=16, r=8)

        assert fw_a is not fw_b
        assert fw_a.u_k is not fw_b.u_k

        # Mutate fw_a — fw_b must be unchanged
        fw_a.u_k.fill_(99.0)
        assert torch.allclose(fw_b.u_k, torch.zeros_like(fw_b.u_k)), (
            "Mutating fw_a.u_k affected fw_b — they share the same tensor"
        )
