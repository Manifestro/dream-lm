"""
Stage 2 experiment: train DREAM-LM on tiny Shakespeare.

Downloads tiny-shakespeare dataset, trains the model, logs results,
and verifies perplexity target (< 15.0).

Usage:
    PYTHONPATH=. uv run python experiments/stage02_train.py

Output:
    data/tiny-shakespeare.txt    — downloaded corpus
    data/vocab.json              — saved vocabulary
    models/stage02.pt            — trained model weights
    experiments/stage02_train/
        results/loss_curve.png   — training curve
        notes.md                 — experiment notes
"""

import json
import math
import os
import time
from pathlib import Path

import torch
import yaml

from dream_lm.core.tokenizer import CharTokenizer
from dream_lm.core.model import DREAMLM
from dream_lm.train.loop import CharDataset, train


def download_tiny_shakespeare(data_dir: Path) -> str:
    """Download tiny Shakespeare dataset if not present.

    ~1.1MB character-level corpus from Andrej Karpathy's nanoGPT.
    """
    filepath = data_dir / "tiny-shakespeare.txt"
    if filepath.exists():
        return filepath.read_text(encoding="utf-8")

    url = "https://raw.githubusercontent.com/karpathy/char-rnn/master/data/tinyshakespeare/input.txt"
    print(f"Downloading tiny-shakespeare from {url}...")
    import urllib.request

    urllib.request.urlretrieve(url, filepath)
    text = filepath.read_text(encoding="utf-8")
    print(f"Downloaded: {len(text)} characters, {filepath.stat().st_size / 1e6:.1f} MB")
    return text


def main():
    # Load config
    config_path = Path("configs/stage02.yaml")
    with open(config_path) as f:
        cfg = yaml.safe_load(f)

    data_dir = Path("data")
    data_dir.mkdir(exist_ok=True)

    # Download and prepare data
    text = download_tiny_shakespeare(data_dir)

    # Build tokenizer
    tokenizer = CharTokenizer.from_text(text)
    tokenizer.save(data_dir / "vocab.json")
    print(f"Vocabulary size: {tokenizer.vocab_size}")

    # Encode full text
    token_ids = tokenizer.encode(text)
    print(f"Total tokens: {len(token_ids)}")

    # Split train/val
    val_split = cfg["data"]["val_split"]
    split_idx = int(len(token_ids) * (1 - val_split))
    train_ids = token_ids[:split_idx]
    val_ids = token_ids[split_idx:]

    print(f"Train: {len(train_ids)} tokens, Val: {len(val_ids)} tokens")

    # Create datasets
    seq_len = cfg["training"]["seq_len"]
    train_ds = CharDataset(train_ids, seq_len)
    val_ds = CharDataset(val_ids, seq_len)

    batch_size = cfg["training"]["batch_size"]
    train_loader = torch.utils.data.DataLoader(train_ds, batch_size=batch_size, shuffle=True)
    val_loader = torch.utils.data.DataLoader(val_ds, batch_size=batch_size)

    # Baseline perplexity (untrained model)
    # For random model: perplexity ≈ vocab_size
    baseline_ppl = tokenizer.vocab_size
    print(f"\nBaseline perplexity (random): {baseline_ppl:.1f}")
    print(f"Target perplexity: < {cfg['eval']['target_perplexity']}")

    # Create model
    model_cfg = cfg["model"]
    model = DREAMLM(
        vocab_size=tokenizer.vocab_size,
        d_model=model_cfg["d_model"],
        n_heads=model_cfg["n_heads"],
        n_layers=model_cfg["n_layers"],
        ff_mult=model_cfg["ff_mult"],
        max_seq_len=model_cfg["max_seq_len"],
        activation=model_cfg["activation"],
        tie_embeddings=model_cfg["tie_embeddings"],
    )
    n_params = sum(p.numel() for p in model.parameters())
    print(f"Model parameters: {n_params:,}")

    # Train
    train_cfg = cfg["training"]
    device = train_cfg.get("device", "cpu")
    if device == "cuda" and not torch.cuda.is_available():
        print("CUDA not available, falling back to CPU")
        device = "cpu"

    t0 = time.time()
    history = train(
        model,
        train_loader,
        val_loader,
        lr=train_cfg["lr"],
        weight_decay=train_cfg["weight_decay"],
        grad_clip=train_cfg["grad_clip"],
        epochs=train_cfg["epochs"],
        warmup_steps=train_cfg["warmup_steps"],
        device=device,
        log_every=1,
    )
    training_time = time.time() - t0
    print(f"\nTotal training time: {training_time:.1f}s")

    # Save model
    models_dir = Path("models")
    models_dir.mkdir(exist_ok=True)
    torch.save(model.state_dict(), models_dir / "stage02.pt")
    print(f"Model saved to {models_dir / 'stage02.pt'}")

    # Generate samples
    print("\n--- Generated samples ---")
    model.eval()
    prompt_text = "First Citizen:\nWe are accounted poor"
    prompt_ids = tokenizer.encode(prompt_text)

    temperatures = cfg["eval"]["temperatures"]
    for temp in temperatures:
        torch.manual_seed(42)
        generated_ids = model.generate(
            prompt_ids,
            max_new_tokens=cfg["eval"]["sample_length"],
            temperature=temp,
        )
        generated_text = tokenizer.decode(generated_ids)
        print(f"\nTemperature: {temp}")
        print(generated_text)

    # Final metrics
    final_ppl = history["perplexity"][-1]
    final_loss = history["val_loss"][-1]

    print(f"\n--- Summary ---")
    print(f"Baseline perplexity: {baseline_ppl:.1f}")
    print(f"Final perplexity: {final_ppl:.2f}")
    print(f"Target (< {cfg['eval']['target_perplexity']}): {'PASS' if final_ppl < cfg['eval']['target_perplexity'] else 'FAIL'}")
    print(f"Final val loss: {final_loss:.4f}")
    print(f"Training time: {training_time:.1f}s")

    # Save results as JSON for the report
    results = {
        "baseline_perplexity": baseline_ppl,
        "final_perplexity": final_ppl,
        "final_val_loss": final_loss,
        "training_time_sec": training_time,
        "n_parameters": n_params,
        "history": {
            "train_loss": history["train_loss"],
            "val_loss": history["val_loss"],
            "perplexity": history["perplexity"],
            "lr": history["lr"],
        },
        "target_perplexity": cfg["eval"]["target_perplexity"],
        "target_met": final_ppl < cfg["eval"]["target_perplexity"],
    }

    results_dir = Path("experiments/stage02_train/results")
    results_dir.mkdir(parents=True, exist_ok=True)
    with open(results_dir / "results.json", "w") as f:
        json.dump(results, f, indent=2)

    print(f"\nResults saved to {results_dir / 'results.json'}")


if __name__ == "__main__":
    main()
