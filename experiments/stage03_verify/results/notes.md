# Stage 3 — KV-Cache: Experiment Notes

## What we expected

KV-Cache should produce numerically identical output to full re-computation (greedy decoding), while being faster per token. The speedup should scale with context length: O(1) per token vs O(n) per token for full recomputation.

Implementation should be backward compatible — training forward pass unchanged, all existing tests pass.

## What we got

**Correctness: PASS** — Cached generation output is bit-identical to uncached across 10 different seed/length configurations (seeds 0-9, prompt lengths 1-30).

**Speedup:**
- Short context (32+20): 2.0x
- Medium context (64+32): 2.6x
- Long context (128+32): 3.9x
- Very long context (200+32): 5.3x

Cached time per token is nearly constant (0.80-0.94 ms), while uncached grows linearly (1.65-4.95 ms).

**Tests: 66/66 passed** (50 existing + 16 new)

## Critical bug found and fixed

**Causal mask position error:** When using cache with `seq_len=1` and `total_seq_len=6`, slicing `causal_mask[:1, :6]` gave row 0 of the lower-triangular matrix — which only sees column 0. But the query token is at position 5 (the last position in the accumulated sequence), so it should see all 6 columns.

**Fix:** Slice mask from `total_seq_len - seq_len` instead of 0 when cache is provided.

**Symptom:** Without this fix, cached generation diverged from uncached after 3-4 tokens. The numerical identity test caught it immediately — this is exactly why we test identity, not just "does it run".

## What it means

KV-Cache is verified and production-ready. The speedup scales as expected with context length, and correctness is guaranteed by the identity test.

The cache structure (KVCache dataclass) is designed for Stage 9 extension (fast weights U_K, U_V) — adding fields will require zero API changes.

The PE offset parameter is a small but critical addition: without it, incremental inference would assign wrong positions to new tokens.

## Next steps

Stage 4 (Low-Rank Fast Weights) — inject U_K, U_V matrices into the attention K/V computation. The KVCache will store the fast-weight state alongside K/V.
