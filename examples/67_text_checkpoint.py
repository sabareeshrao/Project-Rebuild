"""Lesson 70: save model and vocabulary in one inference checkpoint."""

from pathlib import Path
from tempfile import TemporaryDirectory

import torch

from rebuild_gpt.config import GPTConfig
from rebuild_gpt.gpt import GPTLanguageModel
from rebuild_gpt.text_checkpoint import load_text_checkpoint, save_text_checkpoint
from rebuild_gpt.text_generation import generate_text
from rebuild_gpt.tokenizer import CharTokenizer

tokenizer = CharTokenizer("a tiny robot saw a star.\n")
model = GPTLanguageModel(GPTConfig(
    tokenizer.vocab_size, 8, 8, 2, 1, attention_impl="fused",
)).eval()

with TemporaryDirectory() as folder:
    path = Path(folder) / "model-and-tokenizer.pt"
    save_text_checkpoint(model, tokenizer, path)
    recovered, recovered_tokenizer = load_text_checkpoint(path)
    print(generate_text(
        recovered, recovered_tokenizer, "a", 10,
        generator=torch.Generator().manual_seed(4),
    ))
print("This model is untrained; saving restores identity, not language quality.")
