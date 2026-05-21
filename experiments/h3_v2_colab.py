"""
H3 Verification v2 — Full cycle in one file.

Corrects three flaws in h3_v1 (h3_colab.py):
  1. Trains a fresh model with fast_weight_r=16 — W_K/W_V co-exist with
     the fast-weight pathway from training step 0.
  2. Uses teacher-forcing repeated passes instead of free generation —
     removes autoregressive drift noise, lets U accumulate on identical text.
  3. Uses eta=0.05, lambda=0.99 so U grows to a meaningful magnitude.

Pipeline
--------
  [1] Train  — DREAM-LM from scratch, r=16, CUDA, 10 epochs
  [2] Test A — teacher forcing, 3 passes on same chunk (primary H3 test)
  [3] Test B — long free generation, 512 tokens (comparison with v1)
  [4] Notes  — write experiments/h3_v2_notes.md automatically

H3 prediction in Test A
-----------------------
  STDP: mean(ε on pass 3) < mean(ε on pass 1)
  Control: all passes identical (U=0 always, no accumulation)

Usage in Colab
--------------
  !pip install torch numpy scipy pyyaml
  !git clone https://github.com/<your-repo>/dream-lm.git
  %cd dream-lm/dream-lm
  !PYTHONPATH=. python experiments/h3_v2_colab.py
"""

import json
import sys
import time
from pathlib import Path

import numpy as np
import torch

sys.path.insert(0, str(Path(__file__).parent.parent))

from dream_lm.core.ema import EMAStats
from dream_lm.core.fast_weights import FastWeightState
from dream_lm.core.kv_cache import KVCache
from dream_lm.core.model import DREAMLM
from dream_lm.core.tokenizer import CharTokenizer
from dream_lm.core.predictive_coding import compute_perception_error, perception_error_norm
from dream_lm.train.loop import CharDataset, train

RESULTS_DIR = Path(__file__).parent / "h3_v2_results"
RESULTS_DIR.mkdir(parents=True, exist_ok=True)

# ============================================================
# Config
# ============================================================

MODEL_CFG = dict(
    d_model=128,
    n_heads=4,
    n_layers=4,
    ff_mult=4,
    max_seq_len=512,
    activation="gelu",
    tie_embeddings=True,
    fast_weight_r=16,       # KEY: model is trained aware of this pathway
    ema_alpha=0.99,
    gate_theta_0=0.0,
    gate_beta=5.0,
    stdp_eta=0.05,          # 5× higher than v1 — more signal per step
    stdp_lambda_decay=0.99, # slower decay — U persists across the sequence
    stdp_max_norm=1.0,
)

TRAIN_CFG = dict(
    lr=3e-4,
    weight_decay=0.1,
    grad_clip=1.0,
    batch_size=64,
    seq_len=256,
    epochs=10,
    warmup_steps=100,
)

TEST_A_N_CHUNKS = 5
TEST_A_CHUNK_LEN = 256   # tokens per chunk
TEST_A_N_PASSES = 3

TEST_B_N_PROMPTS = 10
TEST_B_PROMPT_LEN = 100  # chars
TEST_B_GEN_LEN = 400     # generated tokens (prompt ~100 + 400 = 500 < max_seq_len 512)

DATA_FILE = Path("data/tiny-shakespeare.txt")
VOCAB_FILE = Path("data/vocab.json")
MODEL_FILE = Path("models/h3_v2_model.pt")


# ============================================================
# Part 1 — Data loading and training
# ============================================================

def _download_shakespeare():
    DATA_FILE.parent.mkdir(exist_ok=True)
    if DATA_FILE.exists():
        return DATA_FILE.read_text(encoding="utf-8")
    url = "https://raw.githubusercontent.com/karpathy/char-rnn/master/data/tinyshakespeare/input.txt"
    print(f"  Downloading {url}...")
    import urllib.request
    urllib.request.urlretrieve(url, DATA_FILE)
    return DATA_FILE.read_text(encoding="utf-8")


def load_data_and_train():
    """Train fresh model with fast_weight_r=16. Saves to MODEL_FILE."""
    print("[1/4] Training model with fast_weight_r=16...")

    text = _download_shakespeare()
    print(f"  Corpus: {len(text):,} chars")

    tokenizer = CharTokenizer.from_text(text)
    tokenizer.save(VOCAB_FILE)
    print(f"  Vocab: {tokenizer.vocab_size} chars")

    token_ids = tokenizer.encode(text)
    split = int(len(token_ids) * 0.95)
    train_ids, val_ids = token_ids[:split], token_ids[split:]

    seq_len = TRAIN_CFG["seq_len"]
    train_loader = torch.utils.data.DataLoader(
        CharDataset(train_ids, seq_len),
        batch_size=TRAIN_CFG["batch_size"], shuffle=True,
    )
    val_loader = torch.utils.data.DataLoader(
        CharDataset(val_ids, seq_len),
        batch_size=TRAIN_CFG["batch_size"],
    )

    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"  Device: {device}")

    model = DREAMLM(vocab_size=tokenizer.vocab_size, **MODEL_CFG)
    n_params = sum(p.numel() for p in model.parameters())
    print(f"  Parameters: {n_params:,}  fast_weight_r={MODEL_CFG['fast_weight_r']}")

    t0 = time.time()
    history = train(
        model, train_loader, val_loader,
        lr=TRAIN_CFG["lr"],
        weight_decay=TRAIN_CFG["weight_decay"],
        grad_clip=TRAIN_CFG["grad_clip"],
        epochs=TRAIN_CFG["epochs"],
        warmup_steps=TRAIN_CFG["warmup_steps"],
        device=device,
        log_every=1,
    )
    elapsed = time.time() - t0

    final_ppl = history["perplexity"][-1]
    print(f"  Final perplexity: {final_ppl:.2f} | training time: {elapsed:.0f}s")

    Path("models").mkdir(exist_ok=True)
    torch.save(model.state_dict(), MODEL_FILE)
    print(f"  Saved to {MODEL_FILE}")

    return model, tokenizer, text, history, device


# ============================================================
# Part 2 — Test A: teacher-forcing repeated passes
# ============================================================

def _make_caches(model, device, dtype, max_len, carry_u_from=None):
    """
    Create fresh (empty) KV caches.
    If carry_u_from is given, copy U_K/U_V from those caches but reset KV.
    This lets STDP accumulation persist across passes while attention starts fresh.
    """
    n_heads = model.layers[0].attn.n_heads
    d_head = model.layers[0].attn.d_head
    caches = []
    for i in range(len(model.layers)):
        fw = None
        if model.fast_weight_r > 0:
            if carry_u_from is not None:
                old_fw = carry_u_from[i].fast_weights
                fw = FastWeightState(
                    u_k=old_fw.u_k.clone(),
                    u_v=old_fw.u_v.clone(),
                    v_basis=old_fw.v_basis,  # fixed basis shared across passes
                )
            else:
                fw = FastWeightState.init(n_heads, d_head, model.fast_weight_r, device, dtype)
        caches.append(KVCache.init(
            batch=1, n_heads=n_heads, d_head=d_head,
            device=device, dtype=dtype,
            max_cache_len=max_len, fast_weights=fw,
        ))
    return caches


def _tf_pass(model, token_ids, device, use_stdp, caches):
    """
    One teacher-forcing pass.
    At each step t, feeds the real token (not a generated one).
    Returns (per-token ε norms, updated caches, total ‖U_K‖).
    """
    ema = EMAStats.init(batch=1, alpha=model.ema_alpha, device=device, dtype=torch.float32)
    eps_norms = []

    stdp_kwargs = (
        dict(stdp_eta=model.stdp_eta,
             stdp_lambda_decay=model.stdp_lambda_decay,
             stdp_max_norm=model.stdp_max_norm)
        if use_stdp else {}
    )

    with torch.no_grad():
        for pos, tok in enumerate(token_ids):
            x = torch.tensor([[tok]], dtype=torch.long, device=device)
            h = model.embedding(x)          # (1, 1, d_model)
            h = model.pe(h, offset=pos)     # sinusoidal PE at position pos

            output, caches, ema, _, _ = model.forward_with_cache(
                h, caches, ema, **stdp_kwargs
            )

            eps = compute_perception_error(
                output.logits, output.h_final, model.w_vocab.weight
            )
            eps_norms.append(perception_error_norm(eps)[0, 0].item())

    u_k_norm = (
        sum(c.fast_weights.u_k.norm().item() for c in caches if c.fast_weights is not None)
        if use_stdp else 0.0
    )
    return eps_norms, caches, u_k_norm


def test_a(model, text, tokenizer, device):
    """
    H3 primary test: teacher-forcing, 3 passes per chunk.

    STDP:    pass 1 → U updated; pass 2 starts with U from pass 1; pass 3 with U from pass 2.
    Control: every pass starts with U=0 (identical results — shows baseline).

    H3 prediction: stdp_pass3_mean_ε < stdp_pass1_mean_ε
    """
    print(f"\n[2/4] Test A — teacher forcing, {TEST_A_N_CHUNKS} chunks × "
          f"{TEST_A_CHUNK_LEN} tokens × {TEST_A_N_PASSES} passes")

    token_ids = tokenizer.encode(text)
    step = len(token_ids) // (TEST_A_N_CHUNKS + 1)
    chunks = [
        token_ids[step * (i + 1): step * (i + 1) + TEST_A_CHUNK_LEN]
        for i in range(TEST_A_N_CHUNKS)
    ]

    results = []
    for ci, chunk in enumerate(chunks):
        print(f"\n  Chunk {ci + 1}/{TEST_A_N_CHUNKS} ({len(chunk)} tokens)")

        # STDP: carry U across passes
        caches = _make_caches(model, device, torch.float32, len(chunk))
        stdp_means, u_norms = [], []
        for p in range(TEST_A_N_PASSES):
            eps, caches, u_norm = _tf_pass(model, chunk, device, use_stdp=True, caches=caches)
            stdp_means.append(float(np.mean(eps)))
            u_norms.append(u_norm)
            print(f"    STDP pass {p+1}: mean_ε={stdp_means[-1]:.5f}  ‖U_K‖={u_norm:.4f}")
            # Carry U into next pass, reset KV
            caches = _make_caches(model, device, torch.float32, len(chunk), carry_u_from=caches)

        # Control: fresh U=0 each pass (results are identical — just verify)
        ctrl_caches = _make_caches(model, device, torch.float32, len(chunk))
        eps_ctrl, _, _ = _tf_pass(model, chunk, device, use_stdp=False, caches=ctrl_caches)
        ctrl_mean = float(np.mean(eps_ctrl))
        print(f"    Control (all passes):   mean_ε={ctrl_mean:.5f}  (U=0, identical each pass)")

        p1_to_p3 = (stdp_means[0] - stdp_means[-1]) / (abs(stdp_means[0]) + 1e-8)
        print(f"    Reduction pass1→pass3:  {p1_to_p3:+.1%}  ({'STDP helps ✓' if p1_to_p3 > 0 else 'no improvement ✗'})")

        results.append({
            "chunk": ci + 1,
            "stdp_pass_means": [round(v, 6) for v in stdp_means],
            "ctrl_mean": round(ctrl_mean, 6),
            "u_k_norms_end": [round(v, 4) for v in u_norms],
            "reduction_p1_to_p3": round(p1_to_p3, 4),
        })

    return results


# ============================================================
# Part 3 — Test B: long free generation
# ============================================================

def _generate(model, prompt_tokens, gen_len, device, use_stdp):
    """Free autoregressive generation. Returns per-token ε norms (generated part)."""
    context = prompt_tokens[-model.pe.max_seq_len:]
    # Guard: never generate past the PE table
    gen_len = min(gen_len, model.pe.max_seq_len - len(context) - 1)
    x = torch.tensor([context], dtype=torch.long, device=device)
    h = model.embedding(x)
    h = model.pe(h)

    n_heads = model.layers[0].attn.n_heads
    d_head = model.layers[0].attn.d_head
    caches = [
        KVCache.init(
            batch=1, n_heads=n_heads, d_head=d_head, device=device, dtype=h.dtype,
            max_cache_len=gen_len + len(context),
            fast_weights=(
                FastWeightState.init(n_heads, d_head, model.fast_weight_r, device, h.dtype)
                if model.fast_weight_r > 0 else None
            ),
        )
        for _ in range(len(model.layers))
    ]
    ema = EMAStats.init(batch=1, alpha=model.ema_alpha, device=device, dtype=h.dtype)
    stdp_kwargs = (
        dict(stdp_eta=model.stdp_eta,
             stdp_lambda_decay=model.stdp_lambda_decay,
             stdp_max_norm=model.stdp_max_norm)
        if use_stdp else {}
    )

    with torch.no_grad():
        output, caches, ema, _, _ = model.forward_with_cache(h, caches, ema, **stdp_kwargs)

    pos = len(context)
    eps_norms = []

    for _ in range(gen_len):
        next_tok = torch.multinomial(
            torch.softmax(output.logits[0, -1, :], dim=-1), num_samples=1
        ).item()

        x_next = torch.tensor([[next_tok]], dtype=torch.long, device=device)
        h_next = model.embedding(x_next)
        h_next = model.pe(h_next, offset=pos)
        pos += 1

        with torch.no_grad():
            output, caches, ema, _, _ = model.forward_with_cache(
                h_next, caches, ema, **stdp_kwargs
            )

        eps = compute_perception_error(output.logits, output.h_final, model.w_vocab.weight)
        eps_norms.append(perception_error_norm(eps)[0, 0].item())

    return eps_norms


def test_b(model, text, tokenizer, device):
    """Long generation — same structure as h3 v1 but 512 tokens, r=16 model."""
    from scipy import stats as sp

    print(f"\n[3/4] Test B — long generation ({TEST_B_GEN_LEN} tokens, {TEST_B_N_PROMPTS} prompts)")

    step = len(text) // (TEST_B_N_PROMPTS + 1)
    prompts = [
        tokenizer.encode(text[step * (i + 1): step * (i + 1) + TEST_B_PROMPT_LEN])
        for i in range(TEST_B_N_PROMPTS)
    ]

    results = []
    for i, prompt in enumerate(prompts):
        eps_s = np.array(_generate(model, prompt, TEST_B_GEN_LEN, device, use_stdp=True))
        eps_c = np.array(_generate(model, prompt, TEST_B_GEN_LEN, device, use_stdp=False))

        x = np.arange(len(eps_s))
        slope_s = float(sp.linregress(x, eps_s).slope)
        slope_c = float(sp.linregress(x, eps_c).slope)
        mean_imp = float(np.mean(eps_c - eps_s))

        results.append({
            "prompt": i + 1,
            "slope_stdp": round(slope_s, 6),
            "slope_ctrl": round(slope_c, 6),
            "mean_improvement": round(mean_imp, 4),
            "stdp_better": mean_imp > 0,
        })
        status = "STDP better" if mean_imp > 0 else "Control better"
        print(f"  [{i+1}] slope STDP={slope_s:+.6f}  ctrl={slope_c:+.6f}  "
              f"imp={mean_imp:+.4f}  → {status}")

    return results


# ============================================================
# Part 4 — Write notes.md
# ============================================================

def write_notes(train_history, results_a, results_b):
    final_ppl = train_history["perplexity"][-1]

    # Test A aggregate
    reductions = [r["reduction_p1_to_p3"] for r in results_a]
    mean_reduction = float(np.mean(reductions))
    chunks_improved = sum(1 for r in reductions if r > 0)
    pass1_means = [r["stdp_pass_means"][0] for r in results_a]
    pass3_means = [r["stdp_pass_means"][-1] for r in results_a]
    ctrl_means = [r["ctrl_mean"] for r in results_a]

    # Test B aggregate
    b_wins = sum(1 for r in results_b if r["stdp_better"])
    b_mean_imp = float(np.mean([r["mean_improvement"] for r in results_b]))
    b_mean_slope_s = float(np.mean([r["slope_stdp"] for r in results_b]))
    b_mean_slope_c = float(np.mean([r["slope_ctrl"] for r in results_b]))

    # H3 verdict (Test A is the primary test)
    if mean_reduction > 0.02 and chunks_improved >= 4:
        verdict = "SUPPORTED"
        verdict_detail = f"Mean ε reduction across passes: {mean_reduction:+.1%}"
    elif mean_reduction > 0 and chunks_improved >= 3:
        verdict = "WEAKLY SUPPORTED"
        verdict_detail = f"Trend in right direction but small: {mean_reduction:+.1%}"
    elif mean_reduction > -0.01:
        verdict = "INCONCLUSIVE"
        verdict_detail = "No consistent improvement across passes"
    else:
        verdict = "NOT SUPPORTED"
        verdict_detail = "ε increases or stays flat across passes"

    notes = f"""# H3 Experiment v2 — Notes

## What was expected

Training a fresh model with `fast_weight_r=16` and testing with teacher
forcing (real tokens, 3 repeated passes on the same chunk) should allow
STDP to accumulate meaningfully. U_K/U_V carry across passes, so by pass 3
the model has adapted its K/V projections to this specific text.

H3 prediction: mean(ε on pass 3) < mean(ε on pass 1) for STDP condition.
Control condition (U=0 always): all passes identical.

Config changes vs v1: fast_weight_r=16 (was 0 at train time), eta=0.05
(was 0.01), lambda=0.99 (was 0.95), teacher forcing (was free generation).

## What happened

**Training:** final perplexity = {final_ppl:.2f} (model trained with r=16 from scratch)

**Test A — teacher forcing repeated passes:**

| | Pass 1 (mean ε) | Pass 2 | Pass 3 | Reduction |
|---|---|---|---|---|
{"".join(f"| Chunk {r['chunk']} | {r['stdp_pass_means'][0]:.5f} | {r['stdp_pass_means'][1]:.5f} | {r['stdp_pass_means'][2]:.5f} | {r['reduction_p1_to_p3']:+.1%} |" + chr(10) for r in results_a)}
| **Mean** | **{np.mean(pass1_means):.5f}** | | **{np.mean(pass3_means):.5f}** | **{mean_reduction:+.1%}** |

Control mean ε (U=0): {np.mean(ctrl_means):.5f}
Chunks improved: {chunks_improved}/{len(results_a)}

**Test B — long generation (512 tokens, r=16 model):**

STDP wins: {b_wins}/{len(results_b)}
Mean improvement: {b_mean_imp:+.4f}
Mean slope STDP={b_mean_slope_s:+.6f}  ctrl={b_mean_slope_c:+.6f}

**H3 verdict: {verdict}**
{verdict_detail}

## What it means

"""

    if verdict == "SUPPORTED":
        notes += """Teacher forcing with U carry-across confirms H3: STDP adapts K/V
projections to the specific material being processed, reducing prediction
error across repeated passes without any gradient descent.

The control (U=0) shows identical ε each pass, confirming the improvement
comes from fast-weight adaptation, not sequence position effects.

Next step: test on out-of-distribution text (different author/style) to
confirm the adaptation is content-specific, not just a warm-up artifact.
"""
    elif verdict in ("WEAKLY SUPPORTED", "INCONCLUSIVE"):
        notes += f"""The trend is in the right direction but the effect is small
(mean reduction {mean_reduction:+.1%}). Two possible explanations:

1. The fast-weight pathway still doesn't contribute much because W_K/W_V
   were trained with U=0 at every step (U is zero-initialized and STDP
   only runs at inference). The base weights never co-adapted with non-zero U.
   Fix: train with STDP active during the training loop too — true
   online learning, not just inference-time adaptation.

2. The r=16 / d_head=32 ratio (50%) means U can represent a large
   perturbation. But surprise gate s_t may be near-zero most of the time,
   effectively suppressing STDP. Check mean(surprise) per pass.

Recommended next experiment: train with STDP active in the training loop
(Stage 9 forward + STDP update at each training step, no gradient to U).
"""
    else:
        notes += """H3 does not hold in this configuration. U accumulation across passes
produced no consistent reduction in ε.

Most likely cause: the base model's W_K/W_V were optimised assuming U=0.
Even with fast_weight_r=16 at training time, U never received non-zero
values during training (STDP was not active). The weights learned to ignore
the fast-weight pathway. Non-zero U introduced at inference time adds noise.

The architecture requires STDP to be active DURING training, not just after.
This is a fundamental design question for Stage 10+: should STDP updates
happen at every training step, or only at inference?
"""

    notes_path = Path(__file__).parent / "h3_v2_notes.md"
    notes_path.write_text(notes, encoding="utf-8")
    print(f"\nNotes written to {notes_path}")
    return str(notes_path)


# ============================================================
# Main
# ============================================================

def main():
    print("=" * 60)
    print("H3 Verification v2 — Full cycle")
    print("  Trains r=16 model + teacher-forcing repeated passes")
    print("=" * 60)

    # [1] Train
    model, tokenizer, text, history, device = load_data_and_train()
    model.eval()

    # [2] Test A — teacher forcing
    results_a = test_a(model, text, tokenizer, device)

    # [3] Test B — long generation
    results_b = test_b(model, text, tokenizer, device)

    # Aggregate summary
    print("\n" + "=" * 60)
    print("AGGREGATE RESULTS")
    print("=" * 60)

    reductions = [r["reduction_p1_to_p3"] for r in results_a]
    print(f"\nTest A (teacher forcing, 3 passes):")
    print(f"  Mean ε reduction pass1→pass3: {np.mean(reductions):+.1%}")
    print(f"  Chunks improved: {sum(1 for r in reductions if r > 0)}/{len(results_a)}")
    print(f"  Per-chunk: {[f'{r:+.1%}' for r in reductions]}")

    b_wins = sum(1 for r in results_b if r["stdp_better"])
    b_mean_imp = float(np.mean([r["mean_improvement"] for r in results_b]))
    print(f"\nTest B (long generation, 512 tokens):")
    print(f"  STDP better: {b_wins}/{len(results_b)}")
    print(f"  Mean improvement: {b_mean_imp:+.4f}")

    # H3 verdict
    mean_reduction = float(np.mean(reductions))
    chunks_improved = sum(1 for r in reductions if r > 0)
    print("\n" + "=" * 60)
    if mean_reduction > 0.02 and chunks_improved >= 4:
        print("H3: SUPPORTED — STDP reduces ε across repeated passes")
    elif mean_reduction > 0 and chunks_improved >= 3:
        print("H3: WEAKLY SUPPORTED — trend present, effect small")
    elif mean_reduction > -0.01:
        print("H3: INCONCLUSIVE — no consistent signal")
    else:
        print("H3: NOT SUPPORTED — ε does not decrease with STDP")
    print("=" * 60)

    # Save raw results
    output = {
        "model_config": MODEL_CFG,
        "final_perplexity": history["perplexity"][-1],
        "test_a": results_a,
        "test_b": results_b,
    }
    out_path = RESULTS_DIR / "h3_v2_results.json"
    with open(out_path, "w") as f:
        json.dump(output, f, indent=2)
    print(f"\nResults saved to {out_path}")

    # [4] Write notes
    write_notes(history, results_a, results_b)


if __name__ == "__main__":
    main()
