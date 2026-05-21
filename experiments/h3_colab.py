"""
H3 verification on trained model — designed for Google Colab (T4/H100).

Loads a trained model, runs generation with STDP on multiple real prompts
from tiny-shakespeare, and compares against a no-STDP control group.

Usage in Colab:
    !pip install torch numpy scipy
    !git clone https://github.com/<your-repo>/dream-lm.git
    %cd dream-lm/dream-lm
    !PYTHONPATH=. python experiments/h3_colab.py

Requires (all produced by Stage 2 training):
    - data/tiny-shakespeare.txt
    - data/vocab.json
    - models/stage02.pt
"""

import json
import sys
import time
from pathlib import Path

import numpy as np
import torch

sys.path.insert(0, str(Path(__file__).parent.parent))

from dream_lm.core.ema import EMAStats
from dream_lm.core.model import DREAMLM
from dream_lm.core.kv_cache import KVCache
from dream_lm.core.fast_weights import FastWeightState
from dream_lm.core.tokenizer import CharTokenizer
from dream_lm.core.predictive_coding import compute_perception_error, perception_error_norm

RESULTS_DIR = Path(__file__).parent / "h3_colab_results"
RESULTS_DIR.mkdir(parents=True, exist_ok=True)

# ============================================================
# Configuration
# ============================================================
NUM_PROMPTS = 10
PROMPT_LENGTH = 100  # characters
GENERATION_LENGTH = 256  # tokens per prompt

STDP_CONFIG = {
    "eta": 0.01,
    "lambda_decay": 0.95,
    "max_norm": 1.0,
}

DATA_FILE = Path("data/tiny-shakespeare.txt")
VOCAB_FILE = Path("data/vocab.json")
MODEL_FILE = Path("models/stage02.pt")


# ============================================================
# Data loading
# ============================================================
def load_text_and_vocab():
    """Load tiny-shakespeare text and vocabulary."""
    if not DATA_FILE.exists():
        raise FileNotFoundError(
            f"{DATA_FILE} not found. Run Stage 2 training first:\n"
            f"  PYTHONPATH=. uv run python experiments/stage02_train.py"
        )
    text = DATA_FILE.read_text(encoding="utf-8")

    if not VOCAB_FILE.exists():
        raise FileNotFoundError(f"{VOCAB_FILE} not found. Run Stage 2 training first.")
    with open(VOCAB_FILE, "r") as f:
        vocab = json.load(f)

    print(f"  Text: {len(text)} characters")
    print(f"  Vocab: {len(vocab)} characters")
    return text, vocab


def extract_prompts(text, vocab, num_prompts, prompt_length):
    """Extract multiple non-overlapping prompts from different parts of text."""
    tokenizer = CharTokenizer(vocab)
    prompts = []
    step = len(text) // (num_prompts + 1)

    for i in range(num_prompts):
        start = step * (i + 1)
        chunk = text[start:start + prompt_length]
        token_ids = tokenizer.encode(chunk)
        if len(token_ids) >= 10:
            prompts.append(token_ids)
            print(f"  Prompt {i+1}: {repr(chunk[:50])}... ({len(token_ids)} tokens)")

    return prompts


# ============================================================
# Model loading
# ============================================================
def load_trained_model(device: torch.device) -> DREAMLM:
    """Load trained model from Stage 2 checkpoint."""
    if not MODEL_FILE.exists():
        raise FileNotFoundError(
            f"{MODEL_FILE} not found. Run Stage 2 training first."
        )

    with open(VOCAB_FILE, "r") as f:
        vocab = json.load(f)
    vocab_size = len(vocab)

    model = DREAMLM(
        vocab_size=vocab_size,
        d_model=128,
        n_heads=4,
        n_layers=4,
        fast_weight_r=16,
        ema_alpha=0.99,
        gate_theta_0=0.0,
        gate_beta=5.0,
        stdp_eta=STDP_CONFIG["eta"],
        stdp_lambda_decay=STDP_CONFIG["lambda_decay"],
        stdp_max_norm=STDP_CONFIG["max_norm"],
    )

    checkpoint = torch.load(MODEL_FILE, map_location=device, weights_only=True)
    if "model_state_dict" in checkpoint:
        state_dict = checkpoint["model_state_dict"]
    else:
        state_dict = checkpoint

    model.load_state_dict(state_dict)
    model.eval()
    model.to(device)

    print(f"  Model loaded: d_model={model.d_model}, layers={len(model.layers)}")
    print(f"  Parameters: {sum(p.numel() for p in model.parameters()):,}")
    return model


# ============================================================
# Generation with/without STDP
# ============================================================
def _init_caches(model, batch, seq_len, device, dtype):
    """Create KVCache list with FastWeightState."""
    n_heads = model.layers[0].attn.n_heads
    d_head = model.layers[0].attn.d_head
    return [
        KVCache.init(
            batch=batch,
            n_heads=n_heads,
            d_head=d_head,
            device=device,
            dtype=dtype,
            max_cache_len=seq_len,
            fast_weights=FastWeightState.init(
                n_heads, d_head, model.fast_weight_r, device, dtype
            ) if model.fast_weight_r > 0 else None,
        )
        for _ in range(len(model.layers))
    ]


def run_generation(
    model: DREAMLM,
    prompt_tokens: list[int],
    max_new_tokens: int,
    device: torch.device,
    stdp_eta: float = 0.0,
) -> dict:
    """
    Run generation with optional STDP, track perception error per token.

    Returns:
        eps_norms: per-token ‖ε_t‖ (prompt + generated)
        surprise_values: s_t per token
        u_k_norms: aggregate ‖U_K‖ across layers, per step
        generated_tokens: list of generated token IDs
    """
    tokens = list(prompt_tokens)
    context = tokens[-model.pe.max_seq_len:]
    x = torch.tensor([context], dtype=torch.long, device=device)

    h = model.embedding(x)
    h = model.pe(h)

    kv_caches = _init_caches(model, batch=1, seq_len=max_new_tokens + len(context), device=device, dtype=h.dtype)
    ema = EMAStats.init(batch=1, alpha=model.ema_alpha, device=device, dtype=h.dtype)

    eps_norms = []
    surprise_values = []
    u_k_norms = []
    generated_tokens = []

    stdp_args = {}
    if stdp_eta > 0:
        stdp_args = {
            "stdp_eta": stdp_eta,
            "stdp_lambda_decay": model.stdp_lambda_decay,
            "stdp_max_norm": model.stdp_max_norm,
        }

    with torch.no_grad():
        output, kv_caches, ema, surprise, _ = model.forward_with_cache(
            h, kv_caches, ema, **stdp_args
        )

        eps_perc = compute_perception_error(output.logits, output.h_final, model.w_vocab.weight)
        eps_norm = perception_error_norm(eps_perc)
        eps_norms.extend(eps_norm[0].tolist())

        if surprise is not None:
            surprise_values.extend(surprise[0].tolist())

        total_u_norm = sum(
            kv_caches[i].fast_weights.u_k.norm().item()
            for i in range(len(model.layers))
            if kv_caches[i].fast_weights is not None
        )
        u_k_norms.append(total_u_norm)

    pos = len(context)

    for step in range(max_new_tokens):
        logits = output.logits[0, -1, :]
        probs = torch.softmax(logits, dim=-1)
        next_token = torch.multinomial(probs, num_samples=1).item()
        tokens.append(next_token)
        generated_tokens.append(next_token)

        x_next = torch.tensor([[next_token]], dtype=torch.long, device=device)
        h_next = model.embedding(x_next)
        h_next = model.pe(h_next, offset=pos)
        pos += 1

        with torch.no_grad():
            output, kv_caches, ema, surprise, _ = model.forward_with_cache(
                h_next, kv_caches, ema, **stdp_args
            )

            eps_perc = compute_perception_error(output.logits, output.h_final, model.w_vocab.weight)
            eps_norm = perception_error_norm(eps_perc)
            eps_norms.append(eps_norm[0, 0].item())

            if surprise is not None:
                surprise_values.append(surprise[0, 0].item())

            total_u_norm = sum(
                kv_caches[i].fast_weights.u_k.norm().item()
                for i in range(len(model.layers))
                if kv_caches[i].fast_weights is not None
            )
            u_k_norms.append(total_u_norm)

    return {
        "eps_norms": eps_norms,
        "surprise_values": surprise_values,
        "u_k_norms": u_k_norms,
        "generated_tokens": generated_tokens,
    }


# ============================================================
# Analysis
# ============================================================
def analyze_prompt(results_stdp, results_no_stdp) -> dict:
    """Analyze single prompt: STDP vs no-STDP comparison."""
    from scipy import stats

    eps_stdp = np.array(results_stdp["eps_norms"])
    eps_no_stdp = np.array(results_no_stdp["eps_norms"])

    # Only compare generated tokens (after prompt)
    prompt_len = len(eps_stdp) - len(results_stdp["generated_tokens"])
    eps_stdp_gen = eps_stdp[prompt_len:]
    eps_no_stdp_gen = eps_no_stdp[prompt_len:]

    # Linear regression with p-value
    x = np.arange(len(eps_stdp_gen))
    slope_stdp, _, r_stdp, p_stdp, _ = stats.linregress(x, eps_stdp_gen)
    slope_no_stdp, _, r_no_stdp, p_no_stdp, _ = stats.linregress(x, eps_no_stdp_gen)

    # Smoothed slope (window=10)
    window = 10
    if len(eps_stdp_gen) > window:
        eps_stdp_smooth = np.convolve(eps_stdp_gen, np.ones(window) / window, mode='valid')
        slope_stdp_smooth = float(np.polyfit(np.arange(len(eps_stdp_smooth)), eps_stdp_smooth, 1)[0])
    else:
        slope_stdp_smooth = slope_stdp

    # STDP improvement: positive = STDP reduced error more than no-STDP
    improvement = eps_no_stdp_gen - eps_stdp_gen
    mean_improvement = float(np.mean(improvement))

    return {
        "slope_stdp": round(float(slope_stdp), 6),
        "slope_no_stdp": round(float(slope_no_stdp), 6),
        "slope_stdp_smooth": round(slope_stdp_smooth, 6),
        "p_value_stdp": round(float(p_stdp), 4),
        "p_value_no_stdp": round(float(p_no_stdp), 4),
        "r_stdp": round(float(r_stdp), 4),
        "r_no_stdp": round(float(r_no_stdp), 4),
        "eps_stdp_first": round(float(eps_stdp_gen[:10].mean()), 4),
        "eps_stdp_last": round(float(eps_stdp_gen[-10:].mean()), 4),
        "eps_no_stdp_first": round(float(eps_no_stdp_gen[:10].mean()), 4),
        "eps_no_stdp_last": round(float(eps_no_stdp_gen[-10:].mean()), 4),
        "mean_improvement": round(mean_improvement, 4),
        "stdp_better": mean_improvement > 0,
        "u_norm_start": round(results_stdp["u_k_norms"][0], 4),
        "u_norm_end": round(results_stdp["u_k_norms"][-1], 4),
    }


def main():
    print("=" * 60)
    print("H3 Verification — STDP on Trained Model")
    print("=" * 60)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"\nDevice: {device}")
    print(f"Prompts: {NUM_PROMPTS}, Generation: {GENERATION_LENGTH} tokens each")
    print(f"STDP: eta={STDP_CONFIG['eta']}, lambda={STDP_CONFIG['lambda_decay']}")

    # Load data
    print("\n[1/4] Loading text and vocabulary...")
    text, vocab = load_text_and_vocab()

    # Extract prompts
    print("\n[2/4] Extracting prompts from dataset...")
    prompts = extract_prompts(text, vocab, NUM_PROMPTS, PROMPT_LENGTH)
    print(f"  Extracted {len(prompts)} prompts")

    # Load model
    print("\n[3/4] Loading trained model...")
    model = load_trained_model(device)

    # Run generation
    print(f"\n[4/4] Running generation ({len(prompts)} prompts x 2 conditions)...")
    t0 = time.time()

    all_analyses = []
    for i, prompt in enumerate(prompts):
        print(f"\n  Prompt {i+1}/{len(prompts)}...")

        # With STDP
        results_stdp = run_generation(
            model, prompt, GENERATION_LENGTH, device,
            stdp_eta=STDP_CONFIG["eta"]
        )

        # Without STDP (control)
        results_no_stdp = run_generation(
            model, prompt, GENERATION_LENGTH, device,
            stdp_eta=0.0
        )

        analysis = analyze_prompt(results_stdp, results_no_stdp)
        all_analyses.append(analysis)

        status = "STDP better" if analysis["stdp_better"] else "No-STDP better"
        print(f"    Slope STDP: {analysis['slope_stdp']:+.6f} (p={analysis['p_value_stdp']})")
        print(f"    Slope no-STDP: {analysis['slope_no_stdp']:+.6f} (p={analysis['p_value_no_stdp']})")
        print(f"    Improvement: {analysis['mean_improvement']:+.4f} -> {status}")

    elapsed = time.time() - t0
    print(f"\n  Total time: {elapsed:.1f}s")

    # Aggregate results
    slopes_stdp = [a["slope_stdp"] for a in all_analyses]
    slopes_no_stdp = [a["slope_no_stdp"] for a in all_analyses]
    improvements = [a["mean_improvement"] for a in all_analyses]
    stdp_wins = sum(1 for a in all_analyses if a["stdp_better"])

    print(f"\n{'=' * 60}")
    print("AGGREGATE RESULTS")
    print(f"{'=' * 60}")
    print(f"\n  STDP slopes:    mean={np.mean(slopes_stdp):+.6f}, std={np.std(slopes_stdp):.6f}")
    print(f"  No-STDP slopes: mean={np.mean(slopes_no_stdp):+.6f}, std={np.std(slopes_no_stdp):.6f}")
    print(f"  STDP better in {stdp_wins}/{len(prompts)} prompts")
    print(f"  Mean improvement: {np.mean(improvements):+.4f} +/- {np.std(improvements):.4f}")

    # H3 verdict
    significant_stdp = sum(1 for a in all_analyses if a["p_value_stdp"] < 0.05 and a["slope_stdp"] < 0)
    print(f"\n  Significant decreasing trend (p<0.05): {significant_stdp}/{len(prompts)} STDP runs")

    print(f"\n{'=' * 60}")
    if stdp_wins >= len(prompts) * 0.6:
        print("H3: SUPPORTED - STDP reduces perception error vs control")
    elif stdp_wins >= len(prompts) * 0.4:
        print("H3: MIXED - STDP helps on some prompts but not others")
    else:
        print("H3: NOT SUPPORTED - STDP does not consistently reduce error")
    print(f"{'=' * 60}")

    # Save results
    output = {
        "config": {
            "num_prompts": len(prompts),
            "generation_length": GENERATION_LENGTH,
            "stdp_config": STDP_CONFIG,
        },
        "aggregate": {
            "slopes_stdp_mean": round(float(np.mean(slopes_stdp)), 6),
            "slopes_no_stdp_mean": round(float(np.mean(slopes_no_stdp)), 6),
            "stdp_wins": stdp_wins,
            "total_prompts": len(prompts),
            "mean_improvement": round(float(np.mean(improvements)), 4),
        },
        "per_prompt": all_analyses,
    }

    output_path = RESULTS_DIR / "h3_results.json"
    with open(output_path, "w") as f:
        json.dump(output, f, indent=2)

    print(f"\nResults saved to {output_path}")


if __name__ == "__main__":
    main()
