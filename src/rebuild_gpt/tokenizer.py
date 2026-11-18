"""A deliberately simple character tokenizer for our early GPT lessons."""

from collections.abc import Iterable


class CharTokenizer:
    """Assign one stable integer ID to every character in the training text."""

    def __init__(self, text: str) -> None:
        if not text:
            raise ValueError("Cannot build a vocabulary from empty text")
        self.vocab = tuple(sorted(set(text)))
        self.stoi = {char: index for index, char in enumerate(self.vocab)}
        self.itos = {index: char for char, index in self.stoi.items()}

    @classmethod
    def from_vocab(cls, vocab: list[str] | tuple[str, ...]) -> "CharTokenizer":
        """Rebuild the exact character-to-ID mapping stored with a model."""
        if not isinstance(vocab, (list, tuple)) or not vocab:
            raise ValueError("vocab must be a non-empty list of characters")
        if any(not isinstance(char, str) or len(char) != 1 for char in vocab):
            raise ValueError("vocab entries must each be one character")
        if tuple(vocab) != tuple(sorted(set(vocab))):
            raise ValueError("vocab must contain unique characters in sorted order")
        return cls("".join(vocab))

    @property
    def vocab_size(self) -> int:
        return len(self.vocab)

    def encode(self, text: str) -> list[int]:
        """Convert characters to token IDs, rejecting unseen characters."""
        try:
            return [self.stoi[char] for char in text]
        except KeyError as exc:
            raise ValueError(f"Character {exc.args[0]!r} is not in vocabulary") from exc

    def decode(self, ids: Iterable[int]) -> str:
        """Convert token IDs back into text."""
        try:
            return "".join(self.itos[index] for index in ids)
        except KeyError as exc:
            raise ValueError(f"Token ID {exc.args[0]!r} is not in vocabulary") from exc
