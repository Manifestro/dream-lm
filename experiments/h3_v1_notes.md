# H3 Experiment v1 — Notes

## What was expected

STDP updates to U_K and U_V should reduce perception error ε^perc during
autoregressive generation compared to a no-STDP control. Expected: STDP
condition shows a negative or flat error slope; no-STDP shows a higher or
positive slope. At least 7/10 prompts should favour STDP for a confident
positive result.

## What happened

**Run:** 10 Shakespeare prompts, 128 tokens each, eta=0.01, lambda=0.95.
Model loaded from `models/stage02.pt` (trained with fast_weight_r=0), then
fast_weight_r=16 injected at inference time only.

| Metric | Value |
|---|---|
| STDP slope mean | +0.000238 |
| No-STDP slope mean | +0.000296 |
| STDP wins | 5/10 |
| Mean improvement | +0.0024 ± 0.0117 |
| Significant decreasing trend (p<0.05) | 0/10 |

Both slopes are **positive** — ε is growing in both conditions. STDP does not
reduce error; it only slightly slows the growth in some prompts.

Verdict: **H3 INCONCLUSIVE** (not falsified, but not confirmed).

## What it means

Three compounding problems made this test insufficient to judge H3:

**Problem 1 — Model never trained with fast weights.**
`stage02.pt` was trained with `fast_weight_r=0`. W_K and W_V learned to
produce correct K/V without any fast-weight contribution. At inference time,
STDP writes non-zero values into a pathway the base weights were never exposed
to, which adds noise rather than useful signal. The model has no "muscle memory"
for using U_K and U_V.

**Problem 2 — Autoregressive drift dominates the ε signal.**
Each generated token feeds back as input. As the sequence grows, the model
drifts away from the training distribution, which independently causes ε to
rise. STDP cannot compensate for this growth with only 128 tokens and eta=0.01.
The measured improvement is swamped by generation noise.

**Problem 3 — U decays before accumulation can matter.**
With lambda=0.95 and only 128 steps, the geometric series saturates quickly.
U stabilises after ~3–5 steps and stops growing. There is not enough
material for adaptive signal to accumulate.

**Next experiment (h3_v2)** corrects all three:
- Trains a fresh model with `fast_weight_r=16` so W_K/W_V co-exist with
  the fast-weight pathway from training step 0.
- Uses **teacher forcing** (real tokens, not generated ones) to remove
  autoregressive noise.
- Runs **3 passes** on the same text, carrying U across passes so it
  accumulates on identical material.
- Increases to eta=0.05, lambda=0.99, seq_len=256 for meaningful U growth.

H3 as stated ("STDP reduces ε without backpropagation") remains a viable
hypothesis. This test only establishes the conditions under which it cannot
be observed.
