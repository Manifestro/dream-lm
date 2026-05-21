"""
Fast-weight state for low-rank K/V augmentation.

Per-head matrices U_K and U_V augment the standard K/V projections:
    K = W_K @ h  +  U_K @ V_basis.T
    V = W_V @ h  +  U_V @ V_basis.T

U_K and U_V are initialized to zero so the augmented path is
bit-identical to the baseline until they are updated (Stage 9 STDP).

V_basis is a fixed random orthonormal matrix shared across steps but
independent per head — each attention head adapts in its own subspace.

Spec §4 — low-rank fast weights:
    U_K^(l,h), U_V^(l,h) ∈ R^(d_head × r)
    V_basis^(h) ∈ R^(d_head × r),  V_basis^T @ V_basis = I
"""

from __future__ import annotations

from dataclasses import dataclass, field

import torch
from torch import Tensor


@dataclass
class FastWeightState:
    """Low-rank fast-weight state for a single transformer layer.

    Contains per-head U_K, U_V matrices and the fixed V_basis.
    All shapes are (n_heads, d_head, r) for head-independent adaptation.

    Attributes:
        u_k: (n_heads, d_head, r) — fast weights for Key augmentation
        u_v: (n_heads, d_head, r) — fast weights for Value augmentation
        v_basis: (n_heads, d_head, r) — fixed orthonormal basis, per head
    """

    u_k: Tensor | None = field(default=None)
    u_v: Tensor | None = field(default=None)
    v_basis: Tensor | None = field(default=None)

    @property
    def is_empty(self) -> bool:
        """True if fast weights are not initialized (None)."""
        return self.u_k is None

    @staticmethod
    def init(
        n_heads: int,
        d_head: int,
        r: int,
        device: torch.device | str = "cpu",
        dtype: torch.dtype = torch.float32,
    ) -> FastWeightState:
        """Create a FastWeightState with zero-initialized U_K, U_V.

        V_basis is generated as a random orthonormal matrix per head
        via QR decomposition. Fixed after creation — not updated by
        gradient descent or STDP (until Stage 9+).

        Zero init ensures the augmentation is a no-op initially:
            U_K @ V_basis.T = 0  →  K = W_K @ h (unchanged)

        Args:
            n_heads: number of attention heads
            d_head: dimension per head
            r: low-rank dimension
            device: device to allocate on
            dtype: dtype for tensors
        """
        # Zero init — no augmentation until U is updated
        u_k = torch.zeros(n_heads, d_head, r, device=device, dtype=dtype)
        u_v = torch.zeros(n_heads, d_head, r, device=device, dtype=dtype)

        # Random orthonormal basis per head via QR decomposition
        v_basis = torch.zeros(n_heads, d_head, r, device=device, dtype=dtype)
        effective_r = min(r, d_head)  # can't have more orthonormal columns than d_head
        for h in range(n_heads):
            # Random matrix → QR → orthonormal columns
            random_mat = torch.randn(d_head, effective_r, device=device, dtype=dtype)
            q, _ = torch.linalg.qr(random_mat)
            v_basis[h, :, :effective_r] = q

        return FastWeightState(u_k=u_k, u_v=u_v, v_basis=v_basis)

    def augment(self, k: Tensor, v: Tensor) -> tuple[Tensor, Tensor]:
        """Augment K and V with fast-weight contribution.

        Computes U_K @ V_basis.T and U_V @ V_basis.T and adds to
        the input K/V tensors. When U=0 this is a no-op.

        Args:
            k: (batch, n_heads, seq_len, d_head) — standard Key projections
            v: (batch, n_heads, seq_len, d_head) — standard Value projections

        Returns:
            k_aug: (batch, n_heads, seq_len, d_head) — augmented K
            v_aug: (batch, n_heads, seq_len, d_head) — augmented V
        """
        # U @ V_basis.T: (n_heads, d_head, r) @ (n_heads, r, d_head) → (n_heads, d_head, d_head)
        # This is a mixing matrix per head that transforms K/V in d_head space.
        fw_k = self.u_k @ self.v_basis.transpose(-2, -1)  # (n_heads, d_head, d_head)
        fw_v = self.u_v @ self.v_basis.transpose(-2, -1)  # (n_heads, d_head, d_head)

        # Apply mixing: k @ fw_k^T over d_head dimension
        # k: (b, n, s, d) @ (n, d, d)^T → (b, n, s, d)
        # Transpose fw_k: (n_heads, d_head, d_head) → (n_heads, d_head, d_head)
        # Contract last dim of k with second-to-last of fw_k
        k_aug = k + torch.einsum("bnse,nde->bnsd", k, fw_k)

        v_aug = v + torch.einsum("bnse,nde->bnsd", v, fw_v)

        return k_aug, v_aug

    def augment_k(self, k: Tensor) -> Tensor:
        """Augment only K with fast-weight contribution.

        Useful when V is already augmented (e.g., retrieved from cache).

        Args:
            k: (batch, n_heads, seq_len, d_head)

        Returns:
            k_aug: (batch, n_heads, seq_len, d_head)
        """
        fw_k = self.u_k @ self.v_basis.transpose(-2, -1)  # (n_heads, d_head, d_head)
        return k + torch.einsum("bnse,nde->bnsd", k, fw_k)

    def augment_v(self, v: Tensor) -> Tensor:
        """Augment only V with fast-weight contribution.

        Args:
            v: (batch, n_heads, seq_len, d_head)

        Returns:
            v_aug: (batch, n_heads, seq_len, d_head)
        """
        fw_v = self.u_v @ self.v_basis.transpose(-2, -1)  # (n_heads, d_head, d_head)
        return v + torch.einsum("bnse,nde->bnsd", v, fw_v)

    def _compute_stdp_delta(
        self,
        eps: Tensor,
        h_or_e: Tensor,
        surprise: Tensor,
    ) -> Tensor:
        """Compute STDP delta_U in r-space without applying it.

        Projects eps and h/e per-head into r-space via V_basis,
        then computes the surprise-gated outer product averaged
        over batch and sequence dimensions.

        Spec §9:
            ε_per_head = eps.view(b, s, n_heads, d_head)
            ε_r        = einsum("bshd,hdr->bshr", ε_per_head, V_basis)
            h_r        = einsum("bshd,hdr->bshr", h_or_e_per_head, V_basis)
            ΔU         = mean_over_bs((surprise · ε_r) ⊗ h_r)

        Args:
            eps: (batch, seq, d_model) — perception error
            h_or_e: (batch, seq, d_model) — h for U_K, e for U_V
            surprise: (batch, seq) — scalar surprise gate

        Returns:
            delta: (n_heads, d_head, r) — averaged STDP update
        """
        batch, seq, d_model = eps.shape
        n_heads, d_head, r = self.v_basis.shape

        assert d_model == n_heads * d_head, (
            f"d_model={d_model} must equal n_heads×d_head={n_heads}×{d_head}={n_heads * d_head}"
        )

        # Split d_model into per-head d_head
        eps_h = eps.view(batch, seq, n_heads, d_head)
        h_h = h_or_e.view(batch, seq, n_heads, d_head)

        # Project eps to r-space via V_basis; keep h in d_head space
        # ΔU = (s · ε_r) ⊗ h_h  →  (r,) ⊗ (d_head,) → (d_head, r)
        eps_r = torch.einsum("bshd,hdr->bshr", eps_h, self.v_basis)  # (b, s, nh, r)

        # Surprise-gated outer product: h_h (b,s,nh,d_head) × gated_eps (b,s,nh,r)
        # surprise: (batch, seq) → (batch, seq, 1, 1) for broadcasting over (nh, r)
        gated_eps = surprise.unsqueeze(-1).unsqueeze(-1) * eps_r  # (b, s, nh, r)
        delta = h_h.unsqueeze(-1) * gated_eps.unsqueeze(-2)  # (b, s, nh, d_head, r)

        # Average over batch and sequence
        return delta.sum(dim=(0, 1)) / (batch * seq)  # (n_heads, d_head, r)

    @torch.no_grad()
    def stdp_update(
        self,
        eps: Tensor,
        h: Tensor,
        e: Tensor,
        surprise: Tensor,
        eta: float = 0.01,
        lambda_decay: float = 0.95,
        max_norm: float = 1.0,
    ) -> None:
        """Apply STDP update to U_K and U_V in-place.

        U_K ← λ·U_K + η·ΔU_K(h)    using hidden state h
        U_V ← λ·U_V + η·ΔU_V(e)    using embedding e
        Then clip both to max_norm (per-head Frobenius norm).

        Wrapped in torch.no_grad() — STDP does not affect gradient flow
        through the base model weights.

        Spec §9:
            ΔU = η · (s · ε_r) ⊗ h_r
            U  ← λ · U + ΔU
            U  ← clip(U, max_norm)

        Args:
            eps: (batch, seq, d_model) — perception error
            h: (batch, seq, d_model) — pre-attention hidden state (for U_K)
            e: (batch, seq, d_model) — input embedding with PE (for U_V)
            surprise: (batch, seq) — scalar surprise gate
            eta: STDP learning rate
            lambda_decay: exponential decay factor (1.0 = no decay)
            max_norm: maximum Frobenius norm per head
        """
        delta_k = self._compute_stdp_delta(eps, h, surprise)
        delta_v = self._compute_stdp_delta(eps, e, surprise)

        # Decay + update
        self.u_k.mul_(lambda_decay).add_(delta_k, alpha=eta)
        self.u_v.mul_(lambda_decay).add_(delta_v, alpha=eta)

        # Max-norm clipping per head (Frobenius norm of d_head × r matrix)
        for u in (self.u_k, self.u_v):
            norms = u.norm(dim=(1, 2), keepdim=True)  # (n_heads, 1, 1)
            clamp_factor = torch.clamp(max_norm / (norms + 1e-8), max=1.0)
            u.mul_(clamp_factor)
