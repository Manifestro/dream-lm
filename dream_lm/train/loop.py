"""
Training loop for DREAM-LM language model.

Spec §12:
    L_LM = -sum_{t=1}^{T} log p(x_{t+1} | x_{1:t})

Optimizer: AdamW with cosine scheduler + warmup.
Gradient clipping prevents explosion in early training.
"""

import math
import time
from pathlib import Path

import torch
import torch.nn as nn
from torch.utils.data import DataLoader, Dataset

from dream_lm.core.model import DREAMLM
from dream_lm.eval.metrics import perplexity


class CharDataset(Dataset):
    """Character-level dataset for autoregressive LM training.

    Splits text into fixed-length chunks for efficient batching.
    Each sample is (input_chunk, target_chunk) where target is shifted
    by one token (next token prediction).

    Args:
        text: full training text
        seq_len: context window size
    """

    def __init__(self, text: list[int], seq_len: int) -> None:
        self.seq_len = seq_len
        # Need seq_len+1 tokens per chunk (input + shifted target)
        chunk_size = seq_len + 1
        n_chunks = len(text) // chunk_size
        self.data = text[: n_chunks * chunk_size]

    def __len__(self) -> int:
        return len(self.data) // (self.seq_len + 1)

    def __getitem__(self, idx: int) -> tuple[torch.Tensor, torch.Tensor]:
        start = idx * (self.seq_len + 1)
        chunk = self.data[start : start + self.seq_len + 1]
        x = torch.tensor(chunk[: self.seq_len], dtype=torch.long)
        y = torch.tensor(chunk[1:], dtype=torch.long)
        return x, y


class CosineWarmupScheduler:
    """Cosine annealing scheduler with linear warmup.

    Warmup prevents instability in early training steps when gradients
    are large and the model hasn't found a good region yet.

    lr(step) = lr_max * (step / warmup_steps)              if step < warmup_steps
    lr(step) = lr_max * (0.5 + 0.5 * cos(pi * (step - warmup) / (total - warmup)))  otherwise
    """

    def __init__(self, warmup_steps: int, total_steps: int, lr_max: float) -> None:
        self.warmup_steps = warmup_steps
        self.total_steps = total_steps
        self.lr_max = lr_max

    def __call__(self, step: int) -> float:
        if step < self.warmup_steps:
            return self.lr_max * (step / self.warmup_steps)
        cos_factor = 0.5 * (
            1 + math.cos(math.pi * (step - self.warmup_steps) / (self.total_steps - self.warmup_steps))
        )
        return self.lr_max * cos_factor

    def get_lr(self, step: int) -> float:
        return self.__call__(step)


def train_epoch(
    model: DREAMLM,
    dataloader: DataLoader,
    optimizer: torch.optim.Optimizer,
    scheduler: CosineWarmupScheduler,
    device: str,
    grad_clip: float = 1.0,
    global_step: int = 0,
) -> tuple[float, int, int]:
    """Train for one epoch.

    Returns:
        (total_loss, num_batches, new_global_step)
    """
    model.train()
    total_loss = 0.0
    n_batches = 0

    for x, y in dataloader:
        x, y = x.to(device), y.to(device)

        optimizer.zero_grad()
        out = model(x)  # ModelOutput(logits=(batch, seq_len, vocab_size), h_final=...)
        logits = out.logits

        loss = nn.functional.cross_entropy(
            logits.view(-1, logits.size(-1)),
            y.view(-1),
        )
        loss.backward()

        # Gradient clipping — prevents explosion in early training
        torch.nn.utils.clip_grad_norm_(model.parameters(), grad_clip)

        optimizer.step()

        lr = scheduler(global_step)
        for param_group in optimizer.param_groups:
            param_group["lr"] = lr

        total_loss += loss.item()
        n_batches += 1
        global_step += 1

    return total_loss, n_batches, global_step


@torch.no_grad()
def evaluate(
    model: DREAMLM,
    dataloader: DataLoader,
    device: str,
) -> float:
    """Compute average loss over validation set.

    Returns:
        mean cross-entropy loss
    """
    model.eval()
    total_loss = 0.0
    n_batches = 0

    for x, y in dataloader:
        x, y = x.to(device), y.to(device)
        out = model(x)
        logits = out.logits
        loss = nn.functional.cross_entropy(
            logits.view(-1, logits.size(-1)),
            y.view(-1),
        )
        total_loss += loss.item()
        n_batches += 1

    return total_loss / max(n_batches, 1)


def train(
    model: DREAMLM,
    train_loader: DataLoader,
    val_loader: DataLoader,
    lr: float = 3e-4,
    weight_decay: float = 0.1,
    grad_clip: float = 1.0,
    epochs: int = 10,
    warmup_steps: int = 100,
    device: str = "cpu",
    log_every: int = 100,
) -> dict:
    """Full training loop.

    Spec §12 — standard LM training with AdamW + cosine scheduler.

    Returns:
        history dict with train_loss, val_loss, perplexity per epoch
    """
    model = model.to(device)

    optimizer = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=weight_decay)

    total_steps = epochs * len(train_loader)
    scheduler = CosineWarmupScheduler(warmup_steps, total_steps, lr)

    history = {"train_loss": [], "val_loss": [], "perplexity": [], "lr": []}
    global_step = 0

    print(f"Training: {epochs} epochs, {len(train_loader)} batches/epoch, lr={lr}")
    print(f"Warmup: {warmup_steps} steps, total steps: {total_steps}")

    for epoch in range(epochs):
        t0 = time.time()
        train_loss, n_batches, global_step = train_epoch(
            model, train_loader, optimizer, scheduler, device, grad_clip, global_step
        )

        val_loss = evaluate(model, val_loader, device)
        val_ppl = math.exp(val_loss)
        current_lr = scheduler(global_step)

        epoch_time = time.time() - t0

        history["train_loss"].append(train_loss / max(n_batches, 1))
        history["val_loss"].append(val_loss)
        history["perplexity"].append(val_ppl)
        history["lr"].append(current_lr)

        if (epoch + 1) % log_every == 0 or epoch == 0:
            print(
                f"Epoch {epoch + 1}/{epochs} | "
                f"train_loss: {history['train_loss'][-1]:.4f} | "
                f"val_loss: {val_loss:.4f} | "
                f"val_ppl: {val_ppl:.2f} | "
                f"lr: {current_lr:.2e} | "
                f"time: {epoch_time:.1f}s"
            )

    print(f"\nFinal perplexity: {history['perplexity'][-1]:.2f} (target: <15.0)")
    return history
