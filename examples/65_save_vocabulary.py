"""Lesson 68: save the vocabulary needed to interpret trained token IDs."""

from pathlib import Path
from tempfile import TemporaryDirectory

from rebuild_gpt.tokenizer import CharTokenizer
from rebuild_gpt.tokenizer_io import load_tokenizer, save_tokenizer

tokenizer = CharTokenizer("a tiny robot saw a star.\n")
with TemporaryDirectory() as directory:
    path = Path(directory) / "vocab.json"
    save_tokenizer(tokenizer, path)
    restored = load_tokenizer(path)
    sample = "a star."
    print("token IDs:", restored.encode(sample))
    print("decoded:", restored.decode(restored.encode(sample)))
    print("identical ID mappings:", restored.stoi == tokenizer.stoi)
