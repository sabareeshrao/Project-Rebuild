"""Lesson 97: run a tiny character GPT on CPU or CUDA."""

import argparse

from rebuild_gpt.text_training import train_char_gpt


parser = argparse.ArgumentParser(description=__doc__)
parser.add_argument("--device", choices=("cpu", "cuda", "auto"), default="cpu")
args = parser.parse_args()

model, tokenizer, result = train_char_gpt(
    "data/tiny_story.txt",
    steps=2, block_size=4, n_embd=8, n_head=2, n_layer=1,
    batch_size=3, eval_interval=1, eval_batches=1,
    device=args.device,
)
print("model device:", next(model.parameters()).device)
print("vocabulary size:", tokenizer.vocab_size)
print("validation loss:", result["validation_after"])
