<div align="center">

# DREAM-LM

**Dynamic Recall and Elastic Adaptive Memory — Language Model**

*A transformer that rewrites its own weights as it reads.*

[![Stage](https://img.shields.io/badge/stage-1%20%2F%2020-4f86f7?style=flat-square)](#roadmap)
[![Tests](https://img.shields.io/badge/tests-24%20passed-2ea44f?style=flat-square)](#stage-1--done)
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

Seven blocks, built and verified one at a time:

```
Token xₜ
  │
  ▼
[1] Embedding + Sinusoidal PE
  │   eₜ = E · one_hot(xₜ) + PEₜ
  │
  ▼
[2] Attention with Fast Weights                        ┐
  │   Qₜ = W_Q · eₜ                                  │  Blocks 1–3 are
  │   Kₜ = W_K · h̃ₜ₋₁  +  U_K · V_basis ᵀ           │  the core loop.
  │   Vₜ = W_V · h̃ₜ₋₁  +  U_V · V_basis ᵀ           │  Everything else
  │   hₜ = softmax(QₜKᵀ / √dₖ + M_causal) · Vₜ      │  plugs into it.
  │                                                   ┘
  ▼
[3] Goal Block
  │   gₜ = (1 − sₜ₋₁)·detach(gₜ₋₁) + sₜ₋₁·MLP_goal(hₜ, εₜ₋₁)
  │   hₜ'= LayerNorm(hₜ + W_goal · gₜ)
  │
  ▼
[4] Prediction  →  ŷₜ = softmax(W_vocab · hₜ')
  │
  ▼
[5] Dual Error Signals
  │   εₜᵖᵉʳᶜ = hₜ' − W_vocab ᵀ · ŷₜ       (perception: did I understand?)
  │   εₜᵉˣᵖʳ = gₜ  − W_proj  · hₜ'        (expression: did I say what I meant?)
  │
  ▼
[6] Vectorized Surprise Gate   sₜ⁽ⁱ⁾ = σ(βᵢ · (|εₜ⁽ⁱ⁾| − θₜ⁽ⁱ⁾))   G=8 groups
  │
  ├──▶ [7] STDP Update  (no gradient descent)
  │         ΔU_K = η · (sₜ ⊙ εₜᵖᵉʳᶜ) · hₜᵀ · V_basis
  │         ΔU_V = η · (sₜ ⊙ εₜᵖᵉʳᶜ) · eₜᵀ · V_basis
  │         U ← clip(λ·U + ΔU,  κ)
  │
  └──▶ [8] LTC Time Constant
            τₜ  = τ_max − (τ_max − τ_min) · sₜ
            h̃ₜ  = (1 − 1/τₜ) · h̃ₜ₋₁  +  (1/τₜ) · hₜ'   ──▶ fed back to Block 2

  [9] Sleep  (triggered when mean surprise over window > ξ)
            W_K^slow += ρ · G_K ⊙ (U_K · V_basis ᵀ)
            U_K, U_V ← 0
```

**Loss:** $\mathcal{L} = \mathcal{L}_{\text{LM}} + \alpha\,\mathcal{L}_{\text{goal}} + \beta\,\mathcal{L}_{\text{surprise}} + \lambda_\beta\|\boldsymbol{\beta}\|_2^2$

Full specification → [manifestro.io](https://manifestro.io)

---

## Roadmap

| Phase | Stages | Theme | Status |
|-------|--------|-------|--------|
| I — Core | 1 – 4 | Transformer, autoregressive LM, KV-cache, low-rank fast weights | **1 ✅** · 2–4 ⬜ |
| II — Predictive Coding | 5 – 8 | Perception error, EMA statistics, surprise gate (scalar → vector) | ⬜ |
| III — Plasticity | 9 – 12 | STDP, homeostasis, LTC time constant, integration test | ⬜ |
| IV — Intentionality | 13 – 15 | Goal block, expression error, joint loss | ⬜ |
| V — Memory | 16 – 17 | Sleep / consolidation, multi-layer stack | ⬜ |
| VI — Scale | 18 – 20 | Language benchmarks, product hypothesis, preprint | ⬜ |

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

## Quick Start

```bash
uv sync
uv run pytest                                          # all 24 tests
uv run pytest tests/test_attention.py -v               # attention module only
PYTHONPATH=. uv run python experiments/stage01_verify.py  # gradcheck + benchmarks
```

---

## Structure

```
dream_lm/core/           # Stage 1: verified transformer blocks
  attention.py           # CausalAttention
  multihead.py           # MultiHeadAttention
  positional_encoding.py # SinusoidalPositionalEncoding
  transformer_layer.py   # TransformerLayer, FeedForward
tests/                   # 24 tests, 3 types per block minimum
experiments/             # verification scripts
configs/                 # YAML hyperparameter configs (Stage 2+)
```

---

## Open Questions

1. Does $\varepsilon^{\text{perc}}$ carry enough signal to drive useful adaptation — or is the vocab projection too narrow a bottleneck? *(tested in Stage 5)*
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
