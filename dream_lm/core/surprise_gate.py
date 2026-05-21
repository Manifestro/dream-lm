"""
Surprise Gate (scalar) — first plasticity gate.

Wraps normalized perception error in a sigmoid to produce a surprise signal
s_t ∈ [0, 1] that controls whether the system should adapt.

Uses ‖ε^perc‖_norm from EMA (Stage 6) — NOT raw norm — because raw is nearly
constant and would produce a constant gate, defeating the purpose.

Spec §7:
    θ_t = θ_0 + γ·H(x_{t-w:t})    # H = local Shannon entropy (Stage 8)
    s_t   = σ(β·(‖ε_t‖ - θ_t))

Stage 7: θ_t = θ_0 (constant), β fixed.
Stage 8: θ_t includes entropy term, per-channel gate.
"""

from __future__ import annotations

import torch
from torch import Tensor


class SurpriseGate:
    """Scalar surprise gate.

    Produces a per-position surprise signal s_t ∈ [0, 1] from normalized
    perception error. Used later to control plasticity (STDP, Stage 9)
    and goal block updates (Stage 13).

    Attributes:
        theta_0: base threshold — error must exceed this to trigger adaptation
        beta: gate temperature — higher means sharper transition
    """

    def __init__(self, theta_0: float = 0.0, beta: float = 5.0) -> None:
        """
        Args:
            theta_0: base threshold (default 0.0 — centered on EMA mean)
            beta: gate temperature (default 5.0 — moderate sharpness)
        """
        self.theta_0 = theta_0
        self.beta = beta

    def compute_threshold(self, token_history: Tensor | None = None) -> float:
        """Compute the adaptive threshold θ_t.

        Stage 7: returns θ_0 (constant).
        Stage 8: adds entropy term γ·H(x_{t-w:t}) when token_history is provided.

        Args:
            token_history: (batch, window) — token IDs for entropy calculation
                (not used in Stage 7, reserved for Stage 8)

        Returns:
            threshold: float — scalar threshold for all positions
        """
        return self.theta_0

    def forward(self, eps_norm: Tensor) -> Tensor:
        """Compute surprise gate from normalized perception error.

        Spec §7:
            s_t = σ(β·(‖ε_t‖_norm - θ_t))

        Args:
            eps_norm: (batch, seq_len) — normalized perception error from EMA

        Returns:
            s_t: (batch, seq_len) — surprise gate values in [0, 1]
        """
        threshold = self.compute_threshold()
        return torch.sigmoid(self.beta * (eps_norm - threshold))


class VectorizedGate:
    """Per-channel-group surprise gate.

    Each of G channel groups gets its own β_i temperature, allowing
    different groups to respond with different sensitivity to surprise.
    Some channels may be "trigger happy" (high β), others conservative.

    Spec §8:
        s_t^(i) = σ(β_i · (|ε_t^(i)|_norm - θ_i))

    Attributes:
        theta_0: base threshold per group — (G,) or scalar
        beta: per-group temperature — (G,)
    """

    def __init__(
        self,
        G: int,
        theta_0: float | Tensor = 0.0,
        beta: float | Tensor = 5.0,
        device: torch.device | str = "cpu",
        dtype: torch.dtype = torch.float32,
    ) -> None:
        """
        Args:
            G: number of channel groups
            theta_0: base threshold — scalar (shared) or (G,) per-group
            beta: gate temperature — scalar (shared) or (G,) per-group
            device: device for parameter tensors
            dtype: dtype for parameter tensors
        """
        self.G = G

        # Normalize theta_0 to (G,) tensor
        if isinstance(theta_0, Tensor):
            self.theta_0 = theta_0.to(device=device, dtype=dtype)
        else:
            self.theta_0 = torch.full((G,), theta_0, device=device, dtype=dtype)

        # Normalize beta to (G,) tensor
        if isinstance(beta, Tensor):
            self.beta = beta.to(device=device, dtype=dtype)
        else:
            self.beta = torch.full((G,), beta, device=device, dtype=dtype)

    def compute_threshold(self) -> Tensor:
        """Compute adaptive threshold per group.

        Returns:
            threshold: (G,) — per-group thresholds
        """
        return self.theta_0  # Stage 15: add entropy term per group

    def forward(self, eps_norm: Tensor) -> Tensor:
        """Compute per-channel surprise gate.

        Spec §8:
            s_t^(i) = σ(β_i · (|ε_t^(i)|_norm - θ_i))

        Args:
            eps_norm: (batch, seq_len, G) — normalized per-group error

        Returns:
            s_t: (batch, seq_len, G) — per-channel surprise values in [0, 1]
        """
        threshold = self.compute_threshold()  # (G,)
        # Broadcast: beta (G,) * (eps_norm (B,S,G) - threshold (G,))
        return torch.sigmoid(self.beta * (eps_norm - threshold))
