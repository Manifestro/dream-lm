# Stage 2 — Autoregressive Language Model

## What we expected

- Perplexity drops from ~65 (random baseline) to < 15.0 on tiny-shakespeare
- Loss decreases monotonically
- Model generates coherent character-level text (dialogue structure, punctuation)
- Temperature sampling works (no NaN, varied output at higher temps)

## What we got

- **Perplexity: 10.72** (target < 15.0) — PASS
- Loss: 3.55 → 2.37 (monotonic decrease, no overfitting: train ≈ val)
- Training time: 202.7s on CPU (d_model=128, 4 layers, 799K params)
- Model learned character-level patterns: dialogue structure, punctuation, capitalization, common letter sequences (th, he, ing)
- At low temperature: stuck in repetitive `\n` patterns (expected for small char-level LM)
- At high temperature: more varied but noisier (also expected)

### Tokenizer design decision

Char-level tokenizer does **not** require `<unk>` — every character from the corpus is always in the vocabulary by construction. ID 0 is the first character alphabetically (newline `\n` for Shakespeare), not a special token. This eliminates the artifact where the model generates `<unk>` as output text.

> Design note: char-level tokenizer doesn't need `<unk>`. ID 0 maps to a real character, not padding. This is a deliberate design choice — no explanation needed later.

## What it means

Stage 2 proves the full pipeline works: tokenizer → model → training → evaluation → generation. The transformer stack from Stage 1 correctly learns an autoregressive language model. 10 epochs with d_model=128 is enough to beat the target by 30%.

The model is still underfitting (train loss ≈ val loss), meaning more epochs or a larger model would push perplexity further. This is fine — Stage 2 is about verifying the pipeline, not about SOTA quality.

Next stage (KV-Cache) will make inference efficient before we add more complexity (fast weights, plasticity, etc).
