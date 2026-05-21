<div align="center">

# DREAM-LM

**Dynamic Recall and Elastic Adaptive Memory — Language Model**

*A transformer that rewrites its own weights as it reads.*

[![Stage](https://img.shields.io/badge/stage-6%20%2F%2020-4f86f7?style=flat-square)](#roadmap)
[![Tests](https://img.shields.io/badge/tests-121%20passed-2ea44f?style=flat-square)](#stage-6--done)
[![Python](https://img.shields.io/badge/python-3.11+-3776ab?style=flat-square)](https://python.org)
[![License](https://img.shields.io/badge/license-MIT-lightgrey?style=flat-square)](LICENSE)
[![Website](https://img.shields.io/badge/manifestro.io-000000?style=flat-square&logo=data:image/svg+xml;base64,)](https://manifestro.io)

</div>

---

Standard language models are static after training. Every token passes through the same frozen weights — the model cannot adjust to what it is currently reading.

DREAM-LM changes that. It injects a pair of low-rank **fast-weight matrices** into the Key and Value projections of attention, and updates them at each token step using a biologically-inspired **STDP rule** — no backward pass, no optimizer. The update is gated by a **Surprise Gate** that measures how much the model failed to predict its own hidden state. High surprise → adapt. Low surprise → hold course.

On top of plasticity, the model maintains a **Goal Block**: a latent intention vector that evolves with the sequence, modulates generation, and is trained to stay aligned with the output. A **Liquid Time Constant** controls context inertia. A **Sleep mechanism** selectively consolidates fast-weight patterns into permanent memory.

> The core bet: a model that knows what it doesn't know, and updates accordingly, should generalize better under distribution shift — without retraining.

---

## Architecture

Seven blocks. The complete forward pass per token, in order:

```python
# --- input ---
e = embed(x) + sinusoidal_pe(pos)

# --- attention with fast weights ---
# K and V are augmented at inference time; U_K, U_V are updated by STDP, not SGD
Q = W_Q @ e
K = W_K @ h_prev  +  U_K @ V_basis.T
V = W_V @ h_prev  +  U_V @ V_basis.T
h = causal_attention(Q, K, V)

# --- goal block ---
# g tracks latent intention; s gates how much it updates on this token
g = (1 - s_prev) * detach(g_prev)  +  s_prev * MLP_goal(h, eps_prev)
h = LayerNorm(h + W_goal @ g)

# --- prediction ---
y = softmax(W_vocab @ h)

# --- dual error: what I missed vs. what I failed to express ---
eps_perc = h - W_vocab.T @ y
eps_expr = g - W_proj @ h
eps      = eps_perc + W_down @ eps_expr

# --- vectorized surprise gate (G=8 channel groups, learnable temperatures) ---
s = sigmoid(beta[group] * (abs(eps) - theta))   # per-channel, shape (d,)

# --- STDP weight update — no backward pass ---
U_K = clip(decay * U_K  +  lr * outer(s * eps_perc, h_prev @ V_basis),  max_norm)
U_V = clip(decay * U_V  +  lr * outer(s * eps_perc, e     @ V_basis),  max_norm)

# --- LTC: inertia scales inversely with surprise ---
tau    = tau_max - (tau_max - tau_min) * s.mean()
h_prev = (1 - 1/tau) * h_prev  +  (1/tau) * h

# --- sleep: consolidate fast weights into slow weights, then reset ---
if running_mean(s) > threshold:
    W_K += rho * attention_gate(U_K) * (U_K @ V_basis.T)
    U_K, U_V = zeros, zeros
```

**Loss:** $\mathcal{L} = \mathcal{L}_{\text{LM}} + \alpha\,\mathcal{L}_{\text{goal}} + \beta\,\mathcal{L}_{\text{surprise}} + \lambda_\beta\|\boldsymbol{\beta}\|_2^2$

Full specification → [manifestro.io](https://manifestro.io)

---

## Roadmap

| Phase | Stages | Theme | Status |
|-------|--------|-------|--------|
| I — Core | 1 – 4 | Transformer, autoregressive LM, KV-cache, low-rank fast weights | **1-4 done** |
| II — Predictive Coding | 5 – 8 | Perception error, EMA statistics, surprise gate (scalar → vector) | **5 done** · 6-8 open |
| III — Plasticity | 9 – 12 | STDP, homeostasis, LTC time constant, integration test | open |
| IV — Intentionality | 13 – 15 | Goal block, expression error, joint loss | open |
| V — Memory | 16 – 17 | Sleep / consolidation, multi-layer stack | open |
| VI — Scale | 18 – 20 | Language benchmarks, product hypothesis, preprint | open |

---

## Stage 1 — Done

Built from scratch, no `nn.MultiheadAttention`:

- `CausalAttention` — single-head, scaled dot-product, causal bool-buffer mask
- `MultiHeadAttention` — parallel heads, concat + W_O projection, causal bool-buffer mask
- `SinusoidalPositionalEncoding` — deterministic, pre-computed buffer
- `FeedForward` — 4× expansion, activation configurable (`"relu"` / `"gelu"`) via config string
- `TransformerLayer` — pre-norm: LN → Attn → residual → LN → FFN → residual

**24 / 24 tests passed** · gradcheck passed (float64, input Jacobian)

Forward pass benchmark (CPU · `torch.no_grad()` · 20-run avg):

| | batch=1 seq=64 d=128 | batch=4 seq=128 d=256 | batch=2 seq=256 d=512 |
|---|---|---|---|
| 1 layer | 0.21 ms | 2.02 ms | 6.21 ms |

Stack baseline for Stage 17 (batch=4 · seq=128 · d=256 · h=8):

| 1 layer | 4 layers | 6 layers |
|---------|----------|----------|
| 3.52 ms | 13.34 ms | 23.15 ms |

---

## Stage 2 — Done

Full autoregressive language model on top of Stage 1 transformer:

- `CharTokenizer` — encode/decode, vocab persistence (JSON), `<unk>` fallback in decode
- `DREAMLM` — embedding → PE → TransformerLayer × 4 → LayerNorm → W_vocab (tied weights)
- `train.loop` — CharDataset, AdamW + cosine warmup, gradient clipping
- `eval.metrics` — perplexity, bits-per-char

**26 / 26 tests passed** · total: 50 / 50 (Stage 1 + Stage 2)

Training on tiny-shakespeare (1.1M chars, 66-char vocab):

| Metric | Value |
|--------|-------|
| Baseline perplexity | 66.0 (random) |
| Final perplexity | **10.75** |
| Target | < 15.0 |
| Training time | 205s (CPU) |
| Parameters | 799,744 |

Loss decreased monotonically from 3.53 → 2.38. Model learned character-level patterns (dialogue structure, punctuation, common sequences).

---

## Stage 3 — Done

KV-Cache for efficient autoregressive generation — O(n) instead of O(n²):

- `KVCache` — dataclass with K/V tensors, append, FIFO truncation (designed for Stage 9 extension)
- `MultiHeadAttention` — incremental forward with optional `kv_cache` parameter
- `TransformerLayer` — cache passthrough
- `DREAMLM` — two-phase `generate()`: prompt processing → incremental token generation
- `SinusoidalPositionalEncoding` — added `offset` parameter for incremental PE

**17 / 17 new tests passed** · total: 67 / 67 (all stages)

Cached generation output is numerically identical to full re-computation (verified across multiple seeds and context lengths).

Inference speedup (CPU · `torch.no_grad()` · 20-run avg):

| Config | Cached (ms/token) | Uncached (ms/token) | Speedup |
|--------|------------------:|--------------------:|--------:|
| short (32+20) | 0.83 | 1.65 | 2.0x |
| medium (64+32) | 0.80 | 2.12 | 2.6x |
| long (128+32) | 0.86 | 3.32 | 3.9x |
| very long (200+32) | 0.94 | 4.95 | 5.3x |

Cached time is nearly constant across context lengths — O(1) per token. Uncached grows linearly — O(n) per token.

---

## Stage 4 — Done

Low-rank fast-weight structure for K/V augmentation — foundation for STDP plasticity:

- `FastWeightState` — per-head U_K, U_V (zero init) + orthonormal V_basis (QR decomposed)
- `KVCache` — added `fast_weights` field (nested state, not separate argument)
- `MultiHeadAttention` — augment K/V before cache append: `K = W_K(h) + U_K @ V_basis.T`
- `DREAMLM` — `fast_weight_r` parameter, auto-init FastWeightState per layer when `r > 0`

**21 / 21 new tests passed** · total: 88 / 88 (all stages)

Baseline identity verified: `fast_weight_r=0` produces bit-identical output to model without fast weights. V_basis orthonormality confirmed per head (V^T V = I). Gradients flow through augment path to U_K/U_V. Each layer holds an independent `FastWeightState` — no shared state across layers.

Fast weights are zero-initialized — augmentation is a no-op until U is updated (Stage 9 STDP). Cache stores augmented K/V, so past tokens reflect what the model knew at that time.

---

## Stage 5 — Done

Perception error (ε^perc) — the first self-prediction signal:

- `ModelOutput` dataclass — `forward()` returns `(logits, h_final)` with backward compat via `__getitem__`
- `compute_perception_error` — ε^perc = h_final - W_vocab^T @ softmax(logits), einsum "vd,bsv->bsd"
- `perception_error_norm` — per-position L2 norm (batch, seq)
- `predictive_coding.py` — new module, works with and without gradients

**16 / 16 new tests passed** · total: 120 / 120 (all stages)

Baseline measurements on untrained model (random input, d_model=128):

| Metric | Value |
|--------|-------|
| Mean ‖ε^perc‖ | 11.31 |
| Std ‖ε^perc‖ | 0.001 (nearly constant) |
| Pearson correlation with CE loss | -0.01 |
| Spearman rank correlation | -0.03 |

ε^perc is nearly constant for a random model — this is expected. The error signal becomes structured after the Goal Block (Stage 13) adds intention modulation, and after EMA normalization (Stage 6) amplifies relative deviations. The baseline measured here serves as the comparison point for Stage 12 (integration test).

---

## Stage 6 — Done

EMA statistics — normalize perception error relative to running background:

- `EMAStats` dataclass — stateful running mean/variance, per-batch-element, autoregressive update
- `forward_with_cache` — accepts `ema_stats`, computes ε^perc → ‖ε^perc‖ → normalized, returns updated state
- `generate()` — creates EMAStats alongside KVCache, carries through entire generation loop
- Configurable `ema_alpha` (0.9 = fast, 0.99 = default, 0.999 = slow)

**17 / 17 new tests passed** · total: 121 / 121 (all stages)

Normalized signal on untrained model (raw ‖ε^perc‖ was nearly constant):

| Metric | Raw ‖ε^perc‖ | Normalized ‖ε^perc‖_norm |
|--------|-------------|------------------------|
| Mean | 11.3087 | 2.1537 |
| Std | 0.0014 | 1.5394 |

**1000× more variance** after normalization — EMA amplifies micro-fluctuations invisible in the raw signal. Constant input converges to ≈ 0, sudden changes produce sharp peaks (> 6σ). First-position artifact (normalized = 10.0) handled by Stage 7.

---

## Quick Start

```bash
uv sync
uv run pytest                                          # all 121 tests
uv run pytest tests/test_stage06.py -v                 # Stage 6 only
PYTHONPATH=. uv run python experiments/stage02_train.py  # train on tiny-shakespeare
PYTHONPATH=. uv run python experiments/stage03_verify.py # KV-cache verification
PYTHONPATH=. uv run python experiments/stage04_verify.py # fast weights verification
PYTHONPATH=. uv run python experiments/stage05_verify.py # perception error baseline
PYTHONPATH=. uv run python experiments/stage06_verify.py # EMA normalization verification
```

---

## Structure

```
dream_lm/core/           # Stage 1-6: transformer blocks + model
  attention.py           # CausalAttention
  multihead.py           # MultiHeadAttention (KV-cache incremental forward, fast-weight augment)
  positional_encoding.py # SinusoidalPositionalEncoding (+ offset)
  transformer_layer.py   # TransformerLayer, FeedForward
  kv_cache.py            # KVCache dataclass (+ fast_weights field, Stage 4)
  fast_weights.py        # FastWeightState dataclass (Stage 4)
  predictive_coding.py   # compute_perception_error, perception_error_norm (Stage 5)
  ema.py                 # EMAStats dataclass (NEW Stage 6)
  tokenizer.py           # CharTokenizer (encode/decode, save/load)
  model.py               # DREAMLM (+ ModelOutput, fast_weight_r, ema_alpha)
dream_lm/train/          # Training infrastructure
  loop.py                # CharDataset, CosineWarmupScheduler, train() (+ ModelOutput support)
dream_lm/eval/           # Metrics
  metrics.py             # perplexity, bits_per_char
tests/                   # 121 tests (24 Stage 1 + 26 Stage 2 + 17 Stage 3 + 21 Stage 4 + 16 Stage 5 + 17 Stage 6)
experiments/             # launch scripts
  stage01_verify.py      # Stage 1: gradcheck + benchmarks
  stage02_train.py       # Train on tiny-shakespeare
  stage03_verify.py      # KV-cache correctness + benchmark
  stage04_verify.py      # Fast weights correctness + baseline identity
  stage05_verify.py      # Perception error baseline + correlation
  stage06_verify.py      # EMA normalization verification (NEW Stage 6)
configs/                 # YAML hyperparameter configs
  stage02.yaml           # Stage 2 config (+ fast_weight_r, ema)
```

---

## Open Questions

1. Does $\varepsilon^{\text{perc}}$ carry enough signal to drive useful adaptation — or is the vocab projection too narrow a bottleneck? *(Stage 5 baseline: correlation ≈ 0 on raw model, will re-test after Stage 12 integration)*
2. Does the Goal Block learn genuine intention, or collapse into a residual connection? *(tested in Stage 13)*
3. Is Sleep necessary over $\lambda$-decay alone? *(tested in Stage 16)*
4. Can $g_t$ be decoded into natural language? *(open)*

---

## References

- Friston (2010). The free-energy principle. *Nature Reviews Neuroscience.*
- Hasani et al. (2021). Liquid Time-constant Networks. *AAAI.*
- Ba et al. (2016). Using Fast Weights to Attend to the Recent Past. *NeurIPS.*
- Vaswani et al. (2017). Attention Is All You Need. *NeurIPS.*
- Miconi et al. (2018). Differentiable Plasticity. *ICML.*

---

## Citation

```bibtex
@misc{karl2026dreamlm,
  title  = {DREAM-LM: Dynamic Recall and Elastic Adaptive Memory Language Model},
  author = {Karl, Bagzhan and {Manifestro Team}},
  year   = {2026},
  url    = {https://manifestro.io}
}
```

---

<div align="center">

**Bagzhan Karl** · Manifestro Team · [manifestro.io](https://manifestro.io)

*One block at a time.*

</div>
