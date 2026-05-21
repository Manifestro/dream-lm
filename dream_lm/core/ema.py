"""
EMA statistics for perception error normalization.

Maintains running mean and variance of the perception error norm across token steps.
The normalized signal distinguishes structural surprise from background noise —
foundation for the Surprise Gate (Stage 7).

Spec §6 — EMA statistics:
    μ_t = α·μ_{t-1} + (1-α)·‖ε_t‖
    σ²_t = α·σ²_{t-1} + (1-α)·(‖ε_t‖ - μ_t)²
    ‖ε_t‖_norm = (‖ε_t‖ - μ_t) / (σ_t + eps)

Per-batch-element state — each sequence in the batch maintains independent statistics.
"""

from __future__ import annotations

from dataclasses import dataclass

import torch
from torch import Tensor


@dataclass
class EMAStats:
    """Exponential moving average statistics for perception error norm.

    Carries running mean (mu) and variance (var) across token steps.
    Updated autoregressively — each position's normalization depends on
    the statistics accumulated up to that point.

    Supports two modes:
    - Scalar (default): mu/var shape (batch,), input (batch, seq_len)
    - Grouped: mu/var shape (batch, G), input (batch, seq_len, G)

    Attributes:
        mu: running mean — (batch,) or (batch, G)
        var: running variance — (batch,) or (batch, G)
        alpha: float — smoothing factor (0.9 = fast, 0.999 = slow)
        eps: float — numerical stability constant for division
    """

    mu: Tensor | None = None
    var: Tensor | None = None
    alpha: float = 0.99
    eps: float = 1e-8

    @property
    def is_grouped(self) -> bool:
        """True if EMA is operating in per-channel-group mode."""
        return self.mu is not None and self.mu.ndim == 2

    @staticmethod
    def init(
        batch: int,
        alpha: float = 0.99,
        device: torch.device | str = "cpu",
        dtype: torch.dtype = torch.float32,
        G: int | None = None,
    ) -> EMAStats:
        """Create EMAStats with zero-initialized mean and variance.

        Args:
            batch: batch size (number of independent sequences)
            alpha: smoothing factor — higher means more smoothing
            device: device to allocate on
            dtype: dtype for tensors
            G: number of channel groups (None = scalar mode)

        Returns:
            EMAStats with mu=0, var=0, ready for update() calls
        """
        if G is not None:
            # Grouped mode: (batch, G)
            mu = torch.zeros(batch, G, device=device, dtype=dtype)
            var = torch.zeros(batch, G, device=device, dtype=dtype)
        else:
            # Scalar mode: (batch,)
            mu = torch.zeros(batch, device=device, dtype=dtype)
            var = torch.zeros(batch, device=device, dtype=dtype)
        return EMAStats(mu=mu, var=var, alpha=alpha)

    def update(self, eps_norm: Tensor) -> Tensor:
        """Update EMA statistics and return normalized perception error.

        Processes the input sequence left-to-right (autoregressive order).
        Each position's normalized value uses the statistics accumulated
        up to and including that position.

        Spec §6, per position t:
            mu_new = alpha * mu + (1 - alpha) * norm_t
            var_new = alpha * var + (1 - alpha) * (norm_t - mu_new)^2
            norm_norm = (norm_t - mu_new) / (sqrt(var_new) + eps)

        Args:
            eps_norm: (batch, seq_len) or (batch, seq_len, G)

        Returns:
            eps_norm_normalized: same shape as input — z-scored norms,
                approximately N(0,1) after warmup
        """
        if self.is_grouped:
            return self._update_grouped(eps_norm)
        else:
            return self._update_scalar(eps_norm)

    def _update_scalar(self, eps_norm: Tensor) -> Tensor:
        """Scalar EMA update — (batch, seq_len) input/output."""
        batch, seq_len = eps_norm.shape
        assert batch == self.mu.shape[0], (
            f"Batch size mismatch: eps_norm has batch={batch}, "
            f"EMAStats has batch={self.mu.shape[0]}"
        )

        normalized = torch.zeros_like(eps_norm)

        for t in range(seq_len):
            norm_t = eps_norm[:, t]  # (batch,)

            # Update mean — Spec §6: μ_t = α·μ_{t-1} + (1-α)·‖ε_t‖
            mu_new = self.alpha * self.mu + (1 - self.alpha) * norm_t

            # Update variance — Spec §6: σ²_t = α·σ²_{t-1} + (1-α)·(‖ε_t‖ - μ_t)²
            var_new = self.alpha * self.var + (1 - self.alpha) * (norm_t - mu_new) ** 2

            # Normalize — Spec §6: ‖ε_t‖_norm = (‖ε_t‖ - μ_t) / (σ_t + eps)
            sigma_new = torch.sqrt(var_new + self.eps)
            normalized[:, t] = (norm_t - mu_new) / sigma_new

            # Carry forward
            self.mu = mu_new
            self.var = var_new

        return normalized

    def _update_grouped(self, eps_norm: Tensor) -> Tensor:
        """Grouped EMA update — (batch, seq_len, G) input/output."""
        batch, seq_len, G = eps_norm.shape
        assert batch == self.mu.shape[0], (
            f"Batch size mismatch: eps_norm has batch={batch}, "
            f"EMAStats has batch={self.mu.shape[0]}"
        )
        assert G == self.mu.shape[1], (
            f"Group count mismatch: eps_norm has G={G}, "
            f"EMAStats has G={self.mu.shape[1]}"
        )

        normalized = torch.zeros_like(eps_norm)

        for t in range(seq_len):
            norm_t = eps_norm[:, t, :]  # (batch, G)

            # Update mean — per group
            mu_new = self.alpha * self.mu + (1 - self.alpha) * norm_t

            # Update variance — per group
            var_new = self.alpha * self.var + (1 - self.alpha) * (norm_t - mu_new) ** 2

            # Normalize — per group
            sigma_new = torch.sqrt(var_new + self.eps)
            normalized[:, t, :] = (norm_t - mu_new) / sigma_new

            # Carry forward
            self.mu = mu_new
            self.var = var_new

        return normalized
