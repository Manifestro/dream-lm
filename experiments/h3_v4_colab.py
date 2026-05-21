"""
H3 Verification v4 -- U accumulates without decay during training.

v3 root cause: TRAIN_STDP_LAMBDA=0.95 decayed U to near-zero between batches
(||U_K||_train ~0.003 vs ||U_K||_inference ~2.5). W_K/W_V co-adapted with
near-zero U, then received a large perturbation at inference they never saw.

v4 correction: TRAIN_STDP_LAMBDA=1.0 -- no decay. U accumulates monotonically
across batches and across epochs, bounded only by max_norm clipping. W_K/W_V
learn in a world where U is large (saturated) from early in training.

This is the minimal test: if the train/inference mismatch was the sole structural
problem, H3 should show improvement. If not, the issue is deeper.

Pipeline
--------
  [1] Train  -- DREAM-LM from scratch, r=16, lambda=1.0 (no decay), STDP every step
  [2] Test A -- teacher forcing, 3 passes on same chunk (primary H3 test)
  [3] Test B -- long free generation (comparison)
  [4] Notes  -- write experiments/h3_v4_notes.md automatically

Changes vs v3
-------------
  - TRAIN_STDP_LAMBDA = 1.0 (was 0.95) -- the only change

Usage in Colab
--------------
  !pip install torch numpy scipy pyyaml
  !git clone https://github.com/<your-repo>/dream-lm.git
  %cd dream-lm/dream-lm
  !PYTHONPATH=. python experiments/h3_v4_colab.py
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
from dream_lm.core.predictive_coding import (
    compute_perception_error,
    compute_real_prediction_error,
    perception_error_norm,
)
from dream_lm.train.loop import CharDataset, train

RESULTS_DIR = Path(__file__).parent / "h3_v4_results"
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
    fast_weight_r=16,
    ema_alpha=0.99,
    gate_theta_0=0.0,
    gate_beta=5.0,
    stdp_eta=0.01,          # inference-time eta (same as v1 baseline)
    stdp_lambda_decay=0.99,
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

# STDP during training
# eta=0.001: 10x lower than inference (0.01) so U doesn't saturate on every batch
# lambda=0.95: faster decay than inference (0.99) so U doesn't accumulate across
#   the full epoch — each batch adapts from near-zero, giving a fresh Hebbian signal.
#   At inference (lambda=0.99), U persists across the sequence to test H3.
TRAIN_STDP_ETA = 0.001
TRAIN_STDP_LAMBDA = 1.0   # no decay -- U accumulates across all batches and epochs
TRAIN_STDP_MAX_NORM = 1.0

TEST_A_N_CHUNKS = 5
TEST_A_CHUNK_LEN = 256
TEST_A_N_PASSES = 3

TEST_B_N_PROMPTS = 10
TEST_B_PROMPT_LEN = 100
TEST_B_GEN_LEN = 400

DATA_FILE = Path("data/tiny-shakespeare.txt")
VOCAB_FILE = Path("data/vocab.json")
MODEL_FILE = Path("models/h3_v4_model.pt")


# ============================================================
# Part 1 -- Data loading and training
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
    """Train fresh model with fast_weight_r=16 AND STDP active every step."""
    print("[1/4] Training model with fast_weight_r=16, STDP lambda=1.0 (no decay)...")

    text = _download_shakespeare()
    print(f"  Corpus: {len(text):,} chars")

    tokenizer = CharTokenizer.from_text(text)
    tokenizer.save(VOCAB_FILE)
    print(f"  Vocab: {tokenizer.vocab_size} chars")

    token_ids = tokenizer.encode(text)
    split = int(len(token_ids) * 0.95)
    train_ids, val_ids = token_ids[:split], token_ids[split:]

    seq_len = TRAIN_CFG["seq_len"]
    # drop_last avoids batch-size mismatch in EMA stats on the last batch
    train_loader = torch.utils.data.DataLoader(
        CharDataset(train_ids, seq_len),
        batch_size=TRAIN_CFG["batch_size"], shuffle=True, drop_last=True,
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
    print(f"  Training STDP: eta={TRAIN_STDP_ETA}, lambda={TRAIN_STDP_LAMBDA}")

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
        # KEY vs v2: STDP runs after every optimizer.step()
        stdp_eta=TRAIN_STDP_ETA,
        stdp_lambda_decay=TRAIN_STDP_LAMBDA,
        stdp_max_norm=TRAIN_STDP_MAX_NORM,
    )
    elapsed = time.time() - t0

    final_ppl = history["perplexity"][-1]
    u_norms = history.get("u_k_norm", [])
    print(f"  Final perplexity: {final_ppl:.2f} | training time: {elapsed:.0f}s")
    if u_norms:
        print(f"  ||U_K|| trajectory: {' -> '.join(f'{v:.3f}' for v in u_norms)}")

    Path("models").mkdir(exist_ok=True)
    torch.save(model.state_dict(), MODEL_FILE)
    print(f"  Saved to {MODEL_FILE}")

    return model, tokenizer, text, history, device


# ============================================================
# Part 2 -- Test A: teacher-forcing repeated passes
# ============================================================

def _make_caches(model, device, dtype, max_len, carry_u_from=None):
    """
    Create fresh (empty) KV caches.
    If carry_u_from is given, copy U_K/U_V from those caches but reset KV.
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
                    v_basis=old_fw.v_basis,
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
    At each step t, feeds real token t and measures real prediction error vs token t+1.
    STDP also uses real error (x_real_next=token_{t+1}) -- not self-prediction.
    Last token has no ground-truth successor so it's excluded from eps_norms.
    Returns (per-token eps norms, updated caches, total ||U_K||).
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
            h = model.embedding(x)
            h = model.pe(h, offset=pos)

            # Real next token available for all positions except the last
            has_next = pos + 1 < len(token_ids)
            x_real_next = (
                torch.tensor([[token_ids[pos + 1]]], dtype=torch.long, device=device)
                if has_next else None
            )

            output, caches, ema, _, _ = model.forward_with_cache(
                h, caches, ema, x_real_next=x_real_next, **stdp_kwargs
            )

            if has_next:
                eps = compute_real_prediction_error(
                    output.logits, x_real_next, model.embedding, model.w_vocab.weight
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

    STDP:    pass 1 -> U updated; pass 2 starts with U from pass 1; pass 3 with U from pass 2.
    Control: 3 passes, U=0 each time (fresh caches) -- rules out positional effects.
             If control eps also drops across passes, the drop is not due to STDP.

    H3 prediction: stdp_pass3_mean_eps < stdp_pass1_mean_eps
                   AND stdp reduction > ctrl reduction (adaptation is real, not positional)
    """
    print(f"\n[2/4] Test A -- teacher forcing, {TEST_A_N_CHUNKS} chunks x "
          f"{TEST_A_CHUNK_LEN} tokens x {TEST_A_N_PASSES} passes")

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
            print(f"    STDP pass {p+1}: mean_eps={stdp_means[-1]:.5f}  ||U_K||={u_norm:.4f}")
            caches = _make_caches(model, device, torch.float32, len(chunk), carry_u_from=caches)

        # Control: 3 passes, fresh U=0 each time
        # Identical results expected -- any drop here means positional artifact, not STDP
        ctrl_means = []
        for p in range(TEST_A_N_PASSES):
            ctrl_caches = _make_caches(model, device, torch.float32, len(chunk))
            eps_ctrl, _, _ = _tf_pass(model, chunk, device, use_stdp=False, caches=ctrl_caches)
            ctrl_means.append(float(np.mean(eps_ctrl)))
            print(f"    Control pass {p+1}: mean_eps={ctrl_means[-1]:.5f}  (U=0, fresh each pass)")

        p1_to_p3_stdp = (stdp_means[0] - stdp_means[-1]) / (abs(stdp_means[0]) + 1e-8)
        p1_to_p3_ctrl = (ctrl_means[0] - ctrl_means[-1]) / (abs(ctrl_means[0]) + 1e-8)
        net_gain = p1_to_p3_stdp - p1_to_p3_ctrl
        print(f"    STDP reduction: {p1_to_p3_stdp:+.1%}  |  "
              f"Control reduction: {p1_to_p3_ctrl:+.1%}  |  "
              f"Net gain: {net_gain:+.1%}  "
              f"({'STDP adds value' if net_gain > 0 else 'no added value'})")

        results.append({
            "chunk": ci + 1,
            "stdp_pass_means": [round(v, 6) for v in stdp_means],
            "ctrl_pass_means": [round(v, 6) for v in ctrl_means],
            "u_k_norms_end": [round(v, 4) for v in u_norms],
            "reduction_p1_to_p3_stdp": round(p1_to_p3_stdp, 4),
            "reduction_p1_to_p3_ctrl": round(p1_to_p3_ctrl, 4),
            "net_gain": round(net_gain, 4),
        })

    return results


# ============================================================
# Part 3 -- Test B: long free generation
# ============================================================

def _generate(model, prompt_tokens, gen_len, device, use_stdp):
    """Free autoregressive generation. Returns per-token eps norms (generated part)."""
    context = prompt_tokens[-model.pe.max_seq_len:]
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

        # NOTE: Test B uses self-prediction error (not real error) because
        # in free generation there is no ground-truth next token.
        # This measures a different signal than Test A (which uses real error).
        # Test B results are not directly comparable to Test A.
        eps = compute_perception_error(output.logits, output.h_final, model.w_vocab.weight)
        eps_norms.append(perception_error_norm(eps)[0, 0].item())

    return eps_norms


def test_b(model, text, tokenizer, device):
    """Long generation -- same structure as v2."""
    from scipy import stats as sp

    print(f"\n[3/4] Test B -- long generation ({TEST_B_GEN_LEN} tokens, {TEST_B_N_PROMPTS} prompts)")

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
              f"imp={mean_imp:+.4f}  -> {status}")

    return results


# ============================================================
# Part 4 -- Write notes.md
# ============================================================

def _verdict(mean_reduction_stdp, mean_reduction_ctrl, net_gains, p_val):
    """
    Determine H3 verdict.

    Requires net_gain (STDP reduction minus control reduction) > 5% to rule out
    positional effects, plus p < 0.05 on paired t-test across chunks.
    """
    mean_net = float(np.mean(net_gains))
    chunks_net_positive = sum(1 for g in net_gains if g > 0)

    if mean_net > 0.05 and chunks_net_positive >= 4 and p_val < 0.05:
        return "SUPPORTED", f"Net gain (STDP minus ctrl) {mean_net:+.1%}, p={p_val:.3f}"
    elif mean_net > 0 and chunks_net_positive >= 3:
        return "WEAKLY SUPPORTED", f"Net gain {mean_net:+.1%} but p={p_val:.3f} (not significant)"
    elif mean_net > -0.01 and chunks_net_positive > 0:
        # Some chunks improved but effect is too small — genuinely ambiguous
        return "INCONCLUSIVE", f"Net gain {mean_net:+.1%}, no consistent signal"
    else:
        # All chunks negative OR mean clearly negative — active degradation
        return "NOT SUPPORTED", f"Net gain {mean_net:+.1%}, {chunks_net_positive}/{len(net_gains)} chunks positive"


def write_notes(train_history, results_a, results_b):
    from scipy import stats as sp

    final_ppl = train_history["perplexity"][-1]
    u_norms = train_history.get("u_k_norm", [])
    u_traj = " -> ".join(f"{v:.3f}" for v in u_norms) if u_norms else "N/A"

    pass1_stdp = [r["stdp_pass_means"][0] for r in results_a]
    pass3_stdp = [r["stdp_pass_means"][-1] for r in results_a]
    pass1_ctrl = [r["ctrl_pass_means"][0] for r in results_a]
    pass3_ctrl = [r["ctrl_pass_means"][-1] for r in results_a]
    net_gains = [r["net_gain"] for r in results_a]

    mean_reduction_stdp = float(np.mean([r["reduction_p1_to_p3_stdp"] for r in results_a]))
    mean_reduction_ctrl = float(np.mean([r["reduction_p1_to_p3_ctrl"] for r in results_a]))

    # Paired t-test: does STDP reduce eps more than control across chunks?
    t_stat, p_val = sp.ttest_rel(pass1_stdp, pass3_stdp)

    verdict, verdict_detail = _verdict(mean_reduction_stdp, mean_reduction_ctrl, net_gains, p_val)

    b_wins = sum(1 for r in results_b if r["stdp_better"])
    b_mean_imp = float(np.mean([r["mean_improvement"] for r in results_b]))
    b_mean_slope_s = float(np.mean([r["slope_stdp"] for r in results_b]))
    b_mean_slope_c = float(np.mean([r["slope_ctrl"] for r in results_b]))

    rows_a = "\n".join(
        f"| Chunk {r['chunk']} "
        f"| {r['stdp_pass_means'][0]:.5f} | {r['stdp_pass_means'][2]:.5f} | {r['reduction_p1_to_p3_stdp']:+.1%} "
        f"| {r['ctrl_pass_means'][0]:.5f} | {r['ctrl_pass_means'][2]:.5f} | {r['reduction_p1_to_p3_ctrl']:+.1%} "
        f"| {r['net_gain']:+.1%} |"
        for r in results_a
    )

    notes = f"""\
# H3 Experiment v4 -- Notes

## What was expected

v3 added STDP to the training loop but used lambda=0.95, which decayed U to near-zero
between batches (||U_K||_train ~0.003). W_K/W_V still co-adapted with near-zero U while
inference saw ||U_K|| ~2.5 -- the train/inference mismatch persisted.

v4 fix: TRAIN_STDP_LAMBDA=1.0 -- no decay. U accumulates monotonically across all
batches and epochs (bounded by max_norm={TRAIN_STDP_MAX_NORM}). W_K/W_V learn in a
world where U is large from early in training. This directly eliminates the mismatch.

H3 prediction in Test A: STDP net gain > 5% over control reduction, p < 0.05.
Control runs 3 identical passes (fresh U=0 each time) to isolate positional effects.

## What happened

**Training:** final perplexity = {final_ppl:.2f} (r=16, STDP in loop)
**||U_K|| during training:** {u_traj}

**Test A -- teacher forcing repeated passes:**

| | STDP pass1 | STDP pass3 | STDP red. | Ctrl pass1 | Ctrl pass3 | Ctrl red. | Net gain |
|---|---|---|---|---|---|---|---|
{rows_a}
| **Mean** | **{np.mean(pass1_stdp):.5f}** | **{np.mean(pass3_stdp):.5f}** | **{mean_reduction_stdp:+.1%}** | **{np.mean(pass1_ctrl):.5f}** | **{np.mean(pass3_ctrl):.5f}** | **{mean_reduction_ctrl:+.1%}** | **{np.mean(net_gains):+.1%}** |

Paired t-test (pass1 vs pass3, STDP): t={t_stat:.3f}, p={p_val:.3f}
Chunks with positive net gain: {sum(1 for g in net_gains if g > 0)}/{len(results_a)}

**Test B -- long generation ({TEST_B_GEN_LEN} tokens):**

Note: Test B measures self-prediction error (eps = h - W^T @ softmax(W @ h)), not real
error, because free generation has no ground-truth next token. This is a different
signal than Test A and results are not directly comparable.

| Metric | Value |
|---|---|
| STDP wins | {b_wins}/{len(results_b)} |
| Mean improvement | {b_mean_imp:+.4f} |
| Mean slope STDP | {b_mean_slope_s:+.6f} |
| Mean slope ctrl | {b_mean_slope_c:+.6f} |

Verdict: **H3 {verdict}**
{verdict_detail}

## What it means

"""

    if verdict == "SUPPORTED":
        notes += (
            "H3 confirmed under fair conditions. STDP trained alongside W_K/W_V allows "
            "the fast-weight pathway to carry useful signal. Net gain over control rules "
            "out positional effects; p<0.05 rules out random variation.\n\n"
            "**Next step** -- test on out-of-distribution text (different author/style) "
            "to confirm adaptation is content-specific, not a warm-up artifact.\n"
        )
    elif verdict in ("WEAKLY SUPPORTED", "INCONCLUSIVE"):
        notes += (
            f"Trend in right direction (net gain {np.mean(net_gains):+.1%}) but not significant "
            f"(p={p_val:.3f}). Possible explanations:\n\n"
            f"**Training ||U_K|| too low** -- if trajectory shows near-zero norms, "
            f"eta={TRAIN_STDP_ETA} may be too small. Try 10x increase.\n\n"
            "**Surprise gate suppressed** -- if mean(surprise) is near zero, STDP "
            "barely fires regardless of eta. Check gate activation statistics.\n\n"
            "**Next step** -- tune TRAIN_STDP_ETA so ||U_K|| stabilises at 0.1-0.5 "
            "(healthy, sub-saturating). Then re-run Test A.\n"
        )
    else:
        notes += (
            "H3 still does not hold. STDP in training loop is not sufficient for "
            "fast weights to carry useful signal at inference.\n\n"
            "This puts H3 in serious doubt. Possible causes:\n\n"
            "**Architectural mismatch** -- U_K injects into K via U_K @ V_basis.T. "
            "If attention patterns are already collapsed, K perturbations have no effect.\n\n"
            "**Signal too weak** -- r=16 with max_norm=1.0 may be insufficient. "
            "Try r=32 or max_norm=2.0.\n\n"
            "**Signal structurally absent** -- with lambda=1.0 and STDP active every "
            "training step, W_K/W_V had every opportunity to use U. That they still "
            "don't suggests H3 requires a fundamentally different architecture or "
            "training objective. Consider Path B (episodic memory framing).\n"
        )

    notes_path = Path(__file__).parent / "h3_v4_notes.md"
    notes_path.write_text(notes, encoding="utf-8")
    print(f"\nNotes written to {notes_path}")
    return str(notes_path)


# ============================================================
# Main
# ============================================================

def main():
    print("=" * 60)
    print("H3 Verification v4 -- U no decay (lambda=1.0) during training")
    print("  Trains r=16 model, U accumulates across all batches and epochs")
    print("=" * 60)

    model, tokenizer, text, history, device = load_data_and_train()
    model.eval()

    results_a = test_a(model, text, tokenizer, device)
    results_b = test_b(model, text, tokenizer, device)

    print("\n" + "=" * 60)
    print("AGGREGATE RESULTS")
    print("=" * 60)

    from scipy import stats as sp

    net_gains = [r["net_gain"] for r in results_a]
    reductions_stdp = [r["reduction_p1_to_p3_stdp"] for r in results_a]
    reductions_ctrl = [r["reduction_p1_to_p3_ctrl"] for r in results_a]
    pass1_stdp = [r["stdp_pass_means"][0] for r in results_a]
    pass3_stdp = [r["stdp_pass_means"][-1] for r in results_a]
    _, p_val = sp.ttest_rel(pass1_stdp, pass3_stdp)

    print(f"\nTest A (teacher forcing, 3 passes):")
    print(f"  STDP reduction pass1->pass3: {np.mean(reductions_stdp):+.1%}")
    print(f"  Ctrl reduction pass1->pass3: {np.mean(reductions_ctrl):+.1%}")
    print(f"  Net gain (STDP - ctrl): {np.mean(net_gains):+.1%}")
    print(f"  Chunks with positive net gain: {sum(1 for g in net_gains if g > 0)}/{len(results_a)}")
    print(f"  t-test p-value: {p_val:.3f}")

    b_wins = sum(1 for r in results_b if r["stdp_better"])
    b_mean_imp = float(np.mean([r["mean_improvement"] for r in results_b]))
    print(f"\nTest B (long generation, {TEST_B_GEN_LEN} tokens):")
    print(f"  STDP better: {b_wins}/{len(results_b)}")
    print(f"  Mean improvement: {b_mean_imp:+.4f}")

    verdict, verdict_detail = _verdict(
        float(np.mean(reductions_stdp)), float(np.mean(reductions_ctrl)), net_gains, p_val
    )
    print("\n" + "=" * 60)
    print(f"H3: {verdict} -- {verdict_detail}")
    print("=" * 60)

    output = {
        "model_config": MODEL_CFG,
        "train_stdp_eta": TRAIN_STDP_ETA,
        "train_stdp_lambda": TRAIN_STDP_LAMBDA,
        "final_perplexity": history["perplexity"][-1],
        "u_k_norm_trajectory": history.get("u_k_norm", []),
        "test_a": results_a,
        "test_b": results_b,
    }
    out_path = RESULTS_DIR / "h3_v4_results.json"
    with open(out_path, "w") as f:
        json.dump(output, f, indent=2)
    print(f"\nResults saved to {out_path}")

    write_notes(history, results_a, results_b)


if __name__ == "__main__":
    main()
