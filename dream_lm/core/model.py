"""
DREAM-LM language model.

Full autoregressive model: token embedding → positional encoding →
N transformer layers → LayerNorm → vocabulary projection.

Spec §2, §3, §5:
    e_t = E · one_hot(x_t) + P_t
    h = TransformerStack(e)
    y_hat = softmax(W_vocab · h)

Weight tying: W_vocab = E^T when tie_embeddings=True (standard for
parameter efficiency and better generalization).
"""

import torch
import torch.nn as nn
from dataclasses import dataclass
from torch import Tensor

from dream_lm.core.fast_weights import FastWeightState
from dream_lm.core.kv_cache import KVCache
from dream_lm.core.positional_encoding import SinusoidalPositionalEncoding
from dream_lm.core.transformer_layer import TransformerLayer
from dream_lm.core.ema import EMAStats
from dream_lm.core.surprise_gate import SurpriseGate, VectorizedGate
from dream_lm.core.predictive_coding import (
    compute_perception_error,
    perception_error_norm,
    perception_error_groups,
)


@dataclass
class ModelOutput:
    """Output of DREAMLM forward pass.

    Designed for backward compatibility: existing code that expects logits
    can use out.logits (attribute access) or out[0] (tuple indexing).
    New code gains access to h_final for perception error computation.
    """

    logits: Tensor       # (batch, seq_len, vocab_size)
    h_final: Tensor      # (batch, seq_len, d_model) — after final LayerNorm

    def __getitem__(self, index: int):
        """Support tuple-like unpacking: logits, h_final = model(x)."""
        return (self.logits, self.h_final)[index]


class DREAMLM(nn.Module):
    """Autoregressive language model.

    Architecture:
        input IDs → Embedding → PE → TransformerLayer × N → LayerNorm → W_vocab

    Pre-norm transformer stack (stable for deep configurations).
    Low-rank fast weights (U_K, U_V) augment K/V projections in attention.
    """

    def __init__(
        self,
        vocab_size: int,
        d_model: int = 128,
        n_heads: int = 4,
        n_layers: int = 4,
        ff_mult: int = 4,
        max_seq_len: int = 256,
        activation: str = "gelu",
        tie_embeddings: bool = True,
        fast_weight_r: int = 0,
        ema_alpha: float = 0.99,
        gate_theta_0: float = 0.0,
        gate_beta: float = 5.0,
        G: int = 0,
        stdp_eta: float = 0.01,
        stdp_lambda_decay: float = 0.95,
        stdp_max_norm: float = 1.0,
    ) -> None:
        super().__init__()
        self.d_model = d_model
        self.vocab_size = vocab_size
        self.fast_weight_r = fast_weight_r
        self.ema_alpha = ema_alpha
        self.G = G
        self.stdp_eta = stdp_eta
        self.stdp_lambda_decay = stdp_lambda_decay
        self.stdp_max_norm = stdp_max_norm
        self.gate = SurpriseGate(theta_0=gate_theta_0, beta=gate_beta)

        if G > 0:
            self.vec_gate = VectorizedGate(
                G=G, theta_0=gate_theta_0, beta=gate_beta
            )
        else:
            self.vec_gate = None

        self.embedding = nn.Embedding(vocab_size, d_model)
        self.pe = SinusoidalPositionalEncoding(d_model, max_seq_len)

        self.layers = nn.ModuleList(
            [
                TransformerLayer(d_model, n_heads, ff_mult, max_seq_len, activation)
                for _ in range(n_layers)
            ]
        )

        self.ln_f = nn.LayerNorm(d_model)

        # Vocabulary projection
        self.w_vocab = nn.Linear(d_model, vocab_size, bias=False)

        # Weight tying: W_vocab = E^T
        if tie_embeddings:
            self.w_vocab.weight = self.embedding.weight

        self._init_weights()

    def _init_weights(self) -> None:
        """Xavier uniform for all linear layers, normal for embeddings.

        Standard init for transformer models — prevents early training
        instability from extreme weight scales.
        """
        nn.init.normal_(self.embedding.weight, mean=0.0, std=0.02)

        for name, module in self.named_modules():
            if isinstance(module, nn.Linear):
                # Skip tied weights — already initialized by embedding
                if name == "w_vocab":
                    continue
                nn.init.xavier_uniform_(module.weight)
                if module.bias is not None:
                    nn.init.zeros_(module.bias)

    def forward(self, x: Tensor) -> ModelOutput:
        """
        Args:
            x: (batch, seq_len) — token IDs

        Returns:
            ModelOutput with:
                logits: (batch, seq_len, vocab_size)
                h_final: (batch, seq_len, d_model) — after final LayerNorm
        """
        # Embed + positional encoding
        h = self.embedding(x)  # (batch, seq_len, d_model)
        h = self.pe(h)

        # Transformer stack
        for layer in self.layers:
            h, _ = layer(h)

        # Final LayerNorm + vocabulary projection
        h = self.ln_f(h)
        logits = self.w_vocab(h)  # (batch, seq_len, vocab_size)

        return ModelOutput(logits=logits, h_final=h)

    def forward_with_cache(
        self,
        x: Tensor,
        kv_caches: list[KVCache | None],
        ema_stats: EMAStats | None = None,
        ema_stats_grouped: EMAStats | None = None,
        stdp_eta: float | None = None,
        stdp_lambda_decay: float | None = None,
        stdp_max_norm: float | None = None,
    ) -> tuple[ModelOutput, list[KVCache | None], EMAStats | None, Tensor | None, Tensor | None]:
        """Forward pass with KV-Cache for incremental inference.

        Processes a single token (or short sequence) through the model,
        updating the provided KV-Caches for each layer.
        Optionally updates EMA statistics, computes surprise gate,
        and applies STDP update to fast weights.

        When G > 0, computes both scalar and grouped surprise signals.

        Args:
            x: (batch, seq_len, d_model) — embedded input (already has PE applied)
            kv_caches: list of KVCache (one per layer), or None for each layer
            ema_stats: optional scalar EMAStats for perception error normalization
            ema_stats_grouped: optional grouped EMAStats (G groups) for per-channel EMA
            stdp_eta: STDP learning rate (None = skip STDP)
            stdp_lambda_decay: STDP decay factor (None = skip STDP)
            stdp_max_norm: STDP max norm clipping (None = skip STDP)

        Returns:
            ModelOutput with logits and h_final
            updated_caches: list of updated KVCache objects (same objects, mutated in-place)
            updated_ema: scalar EMAStats if provided, None otherwise
            surprise_scalar: (batch, seq_len) scalar surprise values, or None
            surprise_grouped: (batch, seq_len, G) grouped surprise values, or None
        """
        caches_out: list[KVCache | None] = []
        layer_hiddens: list[Tensor] = []  # h before each layer (for STDP)

        h = x
        for i, layer in enumerate(self.layers):
            # Capture h before layer for STDP (post-residual from previous layer)
            layer_hiddens.append(h)

            cache_in = kv_caches[i] if i < len(kv_caches) else None
            if cache_in is not None:
                h, _, cache_out = layer(h, cache_in)
                caches_out.append(cache_out)
            else:
                h, _ = layer(h)
                caches_out.append(None)

        h = self.ln_f(h)
        logits = self.w_vocab(h)

        output = ModelOutput(logits=logits, h_final=h)

        # EMA update + surprise gate (inference-time only)
        updated_ema = ema_stats
        surprise_scalar: Tensor | None = None
        surprise_grouped: Tensor | None = None
        if ema_stats is not None:
            eps_perc = compute_perception_error(logits, h, self.w_vocab.weight)

            # Scalar path (Stage 5/6 baseline)
            eps_norm = perception_error_norm(eps_perc)  # (batch, seq_len)
            eps_norm_normalized = ema_stats.update(eps_norm)
            surprise_scalar = self.gate.forward(eps_norm_normalized)  # (batch, seq_len)

            # Grouped path (Stage 8)
            if self.G > 0 and self.vec_gate is not None and ema_stats_grouped is not None:
                eps_groups = perception_error_groups(eps_perc, self.G)  # (batch, seq_len, G)
                eps_groups_normalized = ema_stats_grouped.update(eps_groups)
                surprise_grouped = self.vec_gate.forward(eps_groups_normalized)

        # STDP update (Stage 9) — only when all three params provided
        if stdp_eta is not None and stdp_lambda_decay is not None and stdp_max_norm is not None:
            for i, cache in enumerate(caches_out):
                if cache is not None and cache.fast_weights is not None:
                    cache.fast_weights.stdp_update(
                        eps=eps_perc,                     # (batch, seq, d_model)
                        h=layer_hiddens[i],               # (batch, seq, d_model) — h before layer
                        e=x,                              # (batch, seq, d_model) — input embedding
                        surprise=surprise_scalar,         # (batch, seq)
                        eta=stdp_eta,
                        lambda_decay=stdp_lambda_decay,
                        max_norm=stdp_max_norm,
                    )

        return output, caches_out, updated_ema, surprise_scalar, surprise_grouped

    @torch.no_grad()
    def generate(
        self,
        prompt: list[int],
        max_new_tokens: int,
        temperature: float = 1.0,
        top_k: int | None = None,
        stdp_eta: float | None = None,
        stdp_lambda_decay: float | None = None,
        stdp_max_norm: float | None = None,
    ) -> list[int]:
        """Autoregressive text generation with KV-Cache.

        Processes the prompt once to build the initial KV-Cache,
        then generates one token at a time using incremental inference.
        O(n) total instead of O(n²) without cache.

        Spec §5:
            y_hat_t = softmax(W_vocab · h_t)

        Args:
            prompt: list of token IDs to start from
            max_new_tokens: how many tokens to generate
            temperature: controls randomness (0.0 = greedy, >1.0 = more random)
            top_k: if set, only sample from top-k logits

        Returns:
            generated token IDs (prompt + new tokens)
        """
        tokens = list(prompt)

        # --- Phase 1: Process prompt, build initial KV-Cache ---
        context = tokens[-self.pe.max_seq_len:]
        x = torch.tensor([context], dtype=torch.long)

        # Embed + PE for the full prompt
        h = self.embedding(x)  # (1, prompt_len, d_model)
        h = self.pe(h)

        # Initialize caches (one per layer)
        device = self.embedding.weight.device
        n_heads = self.layers[0].attn.n_heads
        d_head = self.layers[0].attn.d_head

        # Each layer gets its own FastWeightState — they must not share the same object
        # because Stage 9 (STDP) updates U_K/U_V independently per layer.
        kv_caches: list[KVCache | None] = [
            KVCache.init(
                batch=1,
                n_heads=n_heads,
                d_head=d_head,
                device=device,
                dtype=h.dtype,
                fast_weights=(
                    FastWeightState.init(n_heads, d_head, self.fast_weight_r, device, h.dtype)
                    if self.fast_weight_r > 0
                    else None
                ),
            )
            for _ in range(len(self.layers))
        ]

        # Run through transformer stack with cache to populate it.
        # The last position's logits give us the prediction for the next token.
        ema_stats = EMAStats.init(
            batch=1, alpha=self.ema_alpha, device=device, dtype=h.dtype
        )
        output, kv_caches, ema_stats, _, _ = self.forward_with_cache(
            h, kv_caches, ema_stats,
            stdp_eta=stdp_eta,
            stdp_lambda_decay=stdp_lambda_decay,
            stdp_max_norm=stdp_max_norm,
        )
        logits = output.logits[0, -1, :]  # (vocab_size,)

        # Track actual token position — needed when FIFO truncation is active
        # (kv_cache.seq_len would be capped at max_cache_len, but PE offset must
        # reflect the true position in the sequence, not the cache size)
        pos = len(context)

        # --- Phase 2: Generate tokens one at a time ---
        for step in range(max_new_tokens):
            # Sample from current logits
            if temperature == 0.0:
                next_token = logits.argmax().item()
            else:
                logits = logits / temperature

                # Top-k filtering (optional)
                if top_k is not None:
                    k = min(top_k, logits.size(-1))
                    indices_to_remove = logits < torch.topk(logits, k)[0][..., -1, None]
                    logits[indices_to_remove] = float("-inf")

                probs = torch.softmax(logits, dim=-1)
                next_token = torch.multinomial(probs, num_samples=1).item()
            tokens.append(next_token)

            if step + 1 < max_new_tokens:
                # Feed the newly generated token through the cache for next prediction
                x_next = torch.tensor([[next_token]], dtype=torch.long, device=device)
                h_next = self.embedding(x_next)  # (1, 1, d_model)
                h_next = self.pe(h_next, offset=pos)
                pos += 1

                logits, kv_caches, ema_stats, _, _ = self.forward_with_cache(
                    h_next, kv_caches, ema_stats,
                    stdp_eta=stdp_eta,
                    stdp_lambda_decay=stdp_lambda_decay,
                    stdp_max_norm=stdp_max_norm,
                )
                logits = logits.logits[0, 0, :]  # (vocab_size,)

        return tokens
