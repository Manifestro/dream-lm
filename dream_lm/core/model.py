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

from dream_lm.core.kv_cache import KVCache
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

    def forward_with_cache(
        self, x: Tensor, kv_caches: list[KVCache | None]
    ) -> tuple[Tensor, list[KVCache]]:
        """Forward pass with KV-Cache for incremental inference.

        Processes a single token (or short sequence) through the model,
        updating the provided KV-Caches for each layer.

        Args:
            x: (batch, seq_len, d_model) — embedded input (already has PE)
            kv_caches: list of KVCache (one per layer), or None for each layer

        Returns:
            logits: (batch, seq_len, vocab_size)
            updated_caches: list of updated KVCache objects
        """
        caches_out: list[KVCache] = []

        h = x
        for i, layer in enumerate(self.layers):
            cache_in = kv_caches[i] if i < len(kv_caches) else None
            if cache_in is not None:
                h, _, cache_out = layer(h, cache_in)
                caches_out.append(cache_out)
            else:
                h, _ = layer(h)
                caches_out.append(None)  # type: ignore[arg-type]

        h = self.ln_f(h)
        logits = self.w_vocab(h)

        return logits, caches_out

    @torch.no_grad()
    def generate(
        self,
        prompt: list[int],
        max_new_tokens: int,
        temperature: float = 1.0,
        top_k: int | None = None,
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
        kv_caches: list[KVCache | None] = [
            KVCache.init(
                batch=1,
                n_heads=self.layers[0].attn.n_heads,
                d_head=self.layers[0].attn.d_head,
                device=device,
                dtype=h.dtype,
            )
            for _ in range(len(self.layers))
        ]

        # Run through transformer stack with cache to populate it.
        # The last position's logits give us the prediction for the next token.
        logits, kv_caches = self.forward_with_cache(h, kv_caches)
        logits = logits[0, -1, :]  # (vocab_size,)

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
                h_next = self.pe(h_next, offset=kv_caches[0].seq_len)

                logits, kv_caches = self.forward_with_cache(h_next, kv_caches)
                logits = logits[0, 0, :]  # (vocab_size,)

        return tokens
