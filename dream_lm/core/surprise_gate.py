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
