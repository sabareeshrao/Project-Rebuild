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
