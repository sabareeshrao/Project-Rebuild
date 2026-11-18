"""Lesson 72: train on a tiny story, reload the bundle, and generate text."""

from pathlib import Path
from tempfile import TemporaryDirectory

import torch

from rebuild_gpt.text_checkpoint import load_text_checkpoint, save_text_checkpoint
from rebuild_gpt.text_generation import generate_text
from rebuild_gpt.text_training import train_char_gpt


def main() -> None:
    torch.manual_seed(7)
    model, tokenizer, metrics = train_char_gpt(
        "data/tiny_story.txt", steps=30, block_size=8,
        n_embd=16, n_head=4, n_layer=2,
    )
    with TemporaryDirectory() as directory:
        path = Path(directory) / "text-gpt.pt"
        save_text_checkpoint(model, tokenizer, path)
        restored_model, restored_tok = load_text_checkpoint(path)
        result = generate_text(
            restored_model, restored_tok, "a", 60,
            temperature=0.8, top_k=5,
            generator=torch.Generator().manual_seed(13),
        )
    print("validation loss before:", round(metrics["validation_before"], 4))
    print("validation loss after: ", round(metrics["validation_after"], 4))
    print("generated text:")
    print(result)


if __name__ == "__main__":
    main()
