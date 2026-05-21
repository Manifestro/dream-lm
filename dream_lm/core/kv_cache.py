"""
KV-Cache dataclass for incremental inference.

Stores K and V tensors per layer for autoregressive generation.
Designed to be extended in later stages (fast weights, LTC, STDP).

Spec §3 — KV-cache enables O(1) per-token inference instead of O(n):
    K_cache = [K_1, ..., K_t]    # accumulated keys
    V_cache = [V_1, ..., V_t]    # accumulated values
    Q_t @ K_cache → attn_weights → V_cache
"""

from __future__ import annotations

from dataclasses import dataclass, field

import torch
from torch import Tensor


@dataclass
class KVCache:
    """KV-Cache for a single transformer layer.

    K and V tensors are accumulated as generation proceeds.
    Optional max_cache_len enables FIFO truncation for long sequences.

    Attributes:
        k: (batch, n_heads, seq_len, d_head) — accumulated keys
        v: (batch, n_heads, seq_len, d_head) — accumulated values
        max_cache_len: if set, cache is truncated to this length (FIFO)
    """

    k: Tensor | None = None
    v: Tensor | None = None
    max_cache_len: int | None = field(default=None, repr=False)

    @property
    def seq_len(self) -> int:
        """Current sequence length stored in cache."""
        if self.k is None:
            return 0
        return self.k.shape[2]

    @property
    def is_empty(self) -> bool:
        """True if cache has not been initialized yet."""
        return self.k is None

    def append(self, k_new: Tensor, v_new: Tensor) -> None:
        """Append new K/V tensors to the cache.

        Args:
            k_new: (batch, n_heads, 1, d_head) — K for the new token
            v_new: (batch, n_heads, 1, d_head) — V for the new token
        """
        if self.k is None:
            self.k = k_new
            self.v = v_new
        else:
            self.k = torch.cat([self.k, k_new], dim=2)
            self.v = torch.cat([self.v, v_new], dim=2)

        # FIFO truncation — prepares for LTC on Stage 11
        if self.max_cache_len is not None and self.seq_len > self.max_cache_len:
            self.k = self.k[:, :, -self.max_cache_len:, :]
            self.v = self.v[:, :, -self.max_cache_len:, :]

    @staticmethod
    def init(
        batch: int,
        n_heads: int,
        d_head: int,
        device: torch.device | str = "cpu",
        dtype: torch.dtype = torch.float32,
        max_cache_len: int | None = None,
    ) -> KVCache:
        """Create an empty KVCache ready for incremental inference.

        Returns a cache with zero-length K/V tensors (shape has seq_len=0).
        This allows torch.cat to work on the first append without None checks.

        Args:
            batch: batch size
            n_heads: number of attention heads
            d_head: dimension per head
            device: device to allocate on
            dtype: dtype for cache tensors
            max_cache_len: optional FIFO truncation limit
        """
        k = torch.empty(batch, n_heads, 0, d_head, device=device, dtype=dtype)
        v = torch.empty(batch, n_heads, 0, d_head, device=device, dtype=dtype)
        return KVCache(k=k, v=v, max_cache_len=max_cache_len)
