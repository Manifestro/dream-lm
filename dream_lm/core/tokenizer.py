"""
Character-level tokenizer with vocab persistence.

Spec §2 — token embedding input:
    e_t = E · one_hot(x_t) + P_t

Provides encode (text → token IDs) and decode (token IDs → text)
with a deterministic vocabulary built from the training corpus.
Vocabulary can be saved/loaded as JSON for reproducibility.
"""

import json
from pathlib import Path


class CharTokenizer:
    """Character-level tokenizer.

    Builds vocabulary from a text corpus, assigning each unique character
    a stable integer ID. encode() raises KeyError for unknown characters
    (all chars in inference text must appear in training corpus). decode()
    uses <unk> as a fallback for missing IDs.

    Vocab is saved/loaded as JSON to ensure training and inference use
    the same mapping.
    """

    def __init__(self, vocab: dict[str, int] | None = None) -> None:
        if vocab is not None:
            self.vocab = vocab
        else:
            self.vocab = {}
        self._id2char = {i: ch for ch, i in self.vocab.items()}

    @classmethod
    def from_text(cls, text: str) -> "CharTokenizer":
        """Build vocabulary from a text corpus.

        Characters are sorted for deterministic ordering.
        IDs start at 0 — no special tokens for char-level
        (all characters from the text are always in vocab).
        """
        chars = sorted(set(text))
        vocab = {ch: i for i, ch in enumerate(chars)}
        return cls(vocab)

    @classmethod
    def load(cls, path: str | Path) -> "CharTokenizer":
        """Load vocabulary from a JSON file."""
        path = Path(path)
        with open(path, "r", encoding="utf-8") as f:
            vocab = json.load(f)
        # JSON keys are strings, but values should be ints
        vocab = {k: int(v) for k, v in vocab.items()}
        return cls(vocab)

    def save(self, path: str | Path) -> None:
        """Save vocabulary to a JSON file."""
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        with open(path, "w", encoding="utf-8") as f:
            json.dump(self.vocab, f, ensure_ascii=False, indent=2)

    @property
    def vocab_size(self) -> int:
        return len(self.vocab)

    def encode(self, text: str) -> list[int]:
        """Convert text to list of token IDs.

        Raises ValueError for characters not in vocab (should not
        happen if vocab was built from the same corpus).
        """
        return [self.vocab[ch] for ch in text]  # will raise if unknown

    def decode(self, ids: list[int]) -> str:
        """Convert list of token IDs back to text."""
        return "".join(self._id2char.get(i, "<unk>") for i in ids)
