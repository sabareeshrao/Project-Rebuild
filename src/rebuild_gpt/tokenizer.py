"""A deliberately simple character tokenizer for our early GPT lessons."""


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
