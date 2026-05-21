"""
Tests for Stage 2 components:
- CharTokenizer: encode/decode roundtrip, vocab persistence
- DREAMLM model: output shape, weight tying, numerical stability
- Training loop: loss decreases, gradient flow
"""

import math
import tempfile
from pathlib import Path

import pytest
import torch

from dream_lm.core.tokenizer import CharTokenizer
from dream_lm.core.model import DREAMLM
from dream_lm.train.loop import CharDataset, CosineWarmupScheduler, train
from dream_lm.eval.metrics import perplexity, bits_per_char


# ============================================================
# CharTokenizer Tests
# ============================================================

class TestCharTokenizer:
    """CharTokenizer: encode/decode, vocab, persistence."""

    def test_vocab_construction(self):
        """Vocabulary built correctly from text."""
        tok = CharTokenizer.from_text("hello world")
        # unique chars: ' ', 'd', 'e', 'h', 'l', 'o', 'r', 'w' = 8
        assert tok.vocab_size == 8
        assert tok.vocab[" "] == 0  # first char alphabetically

    def test_encode_decode_roundtrip(self):
        """encode then decode returns original text."""
        text = "hello world! How are you?"
        tok = CharTokenizer.from_text(text)
        ids = tok.encode(text)
        decoded = tok.decode(ids)
        assert decoded == text

    def test_unknown_character_raises(self):
        """Unknown characters raise ValueError (all chars from corpus are in vocab)."""
        tok = CharTokenizer.from_text("abc")
        with pytest.raises(KeyError):
            tok.encode("abcXYZ")  # X, Y, Z not in vocab

    def test_vocab_save_load(self):
        """Vocabulary persists to/from JSON."""
        tok = CharTokenizer.from_text("test text")
        with tempfile.NamedTemporaryFile(suffix=".json", delete=False) as f:
            tok.save(f.name)
            tok2 = CharTokenizer.load(f.name)
        assert tok.vocab == tok2.vocab
        Path(f.name).unlink()

    def test_deterministic_vocab(self):
        """Same text produces same vocabulary ordering."""
        text = "the quick brown fox"
        tok1 = CharTokenizer.from_text(text)
        tok2 = CharTokenizer.from_text(text)
        assert tok1.vocab == tok2.vocab

    def test_output_shape(self):
        """encode produces list of ints matching input length."""
        tok = CharTokenizer.from_text("hello")
        ids = tok.encode("hello")
        assert len(ids) == 5
        assert all(isinstance(i, int) for i in ids)


# ============================================================
# DREAMLM Model Tests
# ============================================================

class TestDREAMLM:
    """DREAMLM: shape, stability, weight tying, generation."""

    def _make_model(self, vocab_size=65, **kwargs):
        return DREAMLM(vocab_size=vocab_size, **kwargs)

    def test_output_shape(self):
        """Logits have shape (batch, seq_len, vocab_size)."""
        model = self._make_model()
        x = torch.randint(0, 65, (4, 32))
        out = model(x)
        assert out.logits.shape == (4, 32, 65)

    def test_numerical_stability(self):
        """No NaN/Inf on random input."""
        model = self._make_model()
        for _ in range(5):
            x = torch.randint(0, 65, (2, 64))
            out = model(x)
            assert not torch.isnan(out.logits).any()
            assert not torch.isinf(out.logits).any()

    def test_weight_tying(self):
        """When tie_embeddings=True, W_vocab shares weights with embedding."""
        model = self._make_model(tie_embeddings=True)
        assert model.w_vocab.weight is model.embedding.weight

    def test_no_weight_tying(self):
        """When tie_embeddings=False, W_vocab has independent weights."""
        model = self._make_model(tie_embeddings=False)
        assert model.w_vocab.weight is not model.embedding.weight

    def test_batch_independence(self):
        """Each batch element produces independent output."""
        model = self._make_model()
        x = torch.randint(0, 65, (2, 16))
        out = model(x)
        # Different inputs should produce different outputs
        assert not torch.allclose(out.logits[0], out.logits[1], atol=1e-4)

    def test_causal_mask_propagation(self):
        """Model doesn't crash with causal mask (inherited from Stage 1)."""
        model = self._make_model()
        x = torch.randint(0, 65, (1, 1))
        out = model(x)
        assert out.logits.shape == (1, 1, 65)

    def test_long_sequence(self):
        """Model handles sequences up to max_seq_len."""
        model = self._make_model(max_seq_len=256)
        x = torch.randint(0, 65, (1, 256))
        out = model(x)
        assert out.logits.shape == (1, 256, 65)

    def test_sequence_exceeds_max(self):
        """Model raises error when seq_len > max_seq_len."""
        model = self._make_model(max_seq_len=64)
        x = torch.randint(0, 65, (1, 128))
        with pytest.raises(AssertionError):
            model(x)

    def test_gradient_flow(self):
        """Gradients flow through all parameters."""
        model = self._make_model()
        x = torch.randint(0, 65, (2, 16))
        out = model(x)
        loss = out.logits.sum()
        loss.backward()
        for name, param in model.named_parameters():
            assert param.grad is not None, f"No gradient for {name}"

    def test_generation_output_length(self):
        """generate() returns correct number of tokens."""
        model = self._make_model()
        prompt = [0, 1, 2]
        result = model.generate(prompt, max_new_tokens=10, temperature=1.0)
        assert len(result) == len(prompt) + 10

    def test_generation_temperature_zero(self):
        """Temperature=0 uses greedy selection (deterministic, no crash)."""
        model = self._make_model()
        prompt = [0, 1]
        result1 = model.generate(prompt, max_new_tokens=5, temperature=0.0)
        result2 = model.generate(prompt, max_new_tokens=5, temperature=0.0)
        assert len(result1) == 7
        assert result1 == result2  # greedy must be deterministic

    def test_generation_temperature_high(self):
        """High temperature produces varied output."""
        model = self._make_model()
        torch.manual_seed(42)
        prompt = [0]
        result = model.generate(prompt, max_new_tokens=20, temperature=2.0)
        # Should have some diversity (not all same token)
        assert len(set(result)) > 1


# ============================================================
# Training Loop Tests
# ============================================================

class TestTrainingLoop:
    """Training loop: scheduler, loss decrease, dataset."""

    def test_scheduler_warmup(self):
        """LR increases during warmup phase."""
        scheduler = CosineWarmupScheduler(warmup_steps=100, total_steps=1000, lr_max=3e-4)
        lr_0 = scheduler(0)
        lr_50 = scheduler(50)
        lr_100 = scheduler(100)
        assert lr_0 == 0.0
        assert lr_50 < lr_100
        assert abs(lr_100 - 3e-4) < 1e-10

    def test_scheduler_cosine_decay(self):
        """LR decreases after warmup following cosine schedule."""
        scheduler = CosineWarmupScheduler(warmup_steps=100, total_steps=1000, lr_max=3e-4)
        lr_500 = scheduler(500)
        lr_900 = scheduler(900)
        lr_1000 = scheduler(1000)
        assert lr_500 > lr_900 > lr_1000

    def test_char_dataset(self):
        """CharDataset produces correct input/target pairs."""
        # Need seq_len+1 per chunk, so 12 tokens / 5 = 2 chunks
        text = list(range(15))
        ds = CharDataset(text, seq_len=4)
        assert len(ds) == 3  # 15 // 5 = 3
        x, y = ds[0]
        assert x.shape == (4,)
        assert y.shape == (4,)
        assert torch.equal(x, torch.tensor([0, 1, 2, 3]))
        assert torch.equal(y, torch.tensor([1, 2, 3, 4]))

    def test_loss_decreases(self):
        """Training loss decreases on simple synthetic data."""
        torch.manual_seed(42)
        model = DREAMLM(vocab_size=16, d_model=64, n_heads=2, n_layers=1, max_seq_len=32)

        # Simple repeating pattern — easy to learn
        text = (list(range(16)) * 50)
        ds = CharDataset(text, seq_len=16)
        loader = torch.utils.data.DataLoader(ds, batch_size=8, shuffle=True)

        history = train(
            model, loader, loader,
            lr=1e-3, weight_decay=0.0, grad_clip=1.0,
            epochs=5, warmup_steps=10, device="cpu", log_every=1
        )

        # Loss should decrease (may not be monotonic per epoch, but trend is down)
        assert history["train_loss"][-1] < history["train_loss"][0]

    def test_expected_perplexity(self):
        """Perplexity metric returns correct values."""
        # Random predictions over 65-char vocab ≈ perplexity 65
        logits = torch.randn(4, 16, 65)
        targets = torch.randint(0, 65, (4, 16))
        ppl = perplexity(logits, targets)
        # Should be in a reasonable range (not exact 65 due to randomness)
        assert 1.0 < ppl < 500


# ============================================================
# Metrics Tests
# ============================================================

class TestMetrics:
    """Perplexity and bits-per-char calculation."""

    def test_perplexity_perfect_prediction(self):
        """Perfect predictions give perplexity = 1.0."""
        logits = torch.zeros(1, 4, 10)
        logits[:, :, 0] = 100.0  # Near-certain on correct token
        targets = torch.zeros(1, 4, dtype=torch.long)
        ppl = perplexity(logits, targets)
        assert abs(ppl - 1.0) < 0.01

    def test_perplexity_random_prediction(self):
        """Uniform random predictions give perplexity ≈ vocab_size."""
        torch.manual_seed(42)
        vocab_size = 65
        logits = torch.randn(1, 100, vocab_size)
        targets = torch.randint(0, vocab_size, (1, 100))
        ppl = perplexity(logits, targets)
        # Random baseline: perplexity should be around vocab_size
        assert 20 < ppl < 200  # generous range for randomness

    def test_bits_per_char_range(self):
        """bpc is in reasonable range for char-level LM."""
        torch.manual_seed(42)
        logits = torch.randn(1, 64, 65)
        targets = torch.randint(0, 65, (1, 64))
        bpc = bits_per_char(logits, targets)
        # Random char model: bpc ≈ log2(65) ≈ 6.02
        assert 0 < bpc < 10
