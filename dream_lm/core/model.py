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
from torch import Tensor

from dream_lm.core.positional_encoding import SinusoidalPositionalEncoding
from dream_lm.core.transformer_layer import TransformerLayer


class DREAMLM(nn.Module):
    """Autoregressive language model.

    Architecture:
        input IDs → Embedding → PE → TransformerLayer × N → LayerNorm → W_vocab

    Pre-norm transformer stack (stable for deep configurations).
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
    ) -> None:
        super().__init__()
        self.d_model = d_model
        self.vocab_size = vocab_size

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

    def forward(self, x: Tensor) -> Tensor:
        """
        Args:
            x: (batch, seq_len) — token IDs

        Returns:
            logits: (batch, seq_len, vocab_size)
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

        return logits

    @torch.no_grad()
    def generate(
        self,
        prompt: list[int],
        max_new_tokens: int,
        temperature: float = 1.0,
        top_k: int | None = None,
    ) -> list[int]:
        """Autoregressive text generation.

        Uses the full context window (no KV-cache yet — Stage 3).
        Each step re-processes the entire sequence through the model.

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

        for _ in range(max_new_tokens):
            # Truncate context to max_seq_len — PE buffer has fixed size
            context = tokens[-self.pe.max_seq_len:]
            x = torch.tensor([context], dtype=torch.long)
            logits = self(x)  # (1, context_len, vocab_size)

            # Take last token predictions
            logits = logits[0, -1, :]  # (vocab_size,)

            # Greedy decoding for temperature=0, sampling otherwise
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

        return tokens
