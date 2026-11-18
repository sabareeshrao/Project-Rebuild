"""Lesson 102: inspect FP32 and BF16 text-GPT computation."""

import argparse

import torch

from rebuild_gpt.text_training import train_char_gpt


parser = argparse.ArgumentParser(description=__doc__)
parser.add_argument("--device", choices=("cpu", "cuda", "auto"), default="cpu")
parser.add_argument("--precision", choices=("fp32", "bf16", "fp16"), default="bf16")
args = parser.parse_args()

model, tokenizer, metrics = train_char_gpt(
    "data/tiny_story.txt", steps=2, block_size=4, n_embd=8, n_head=2,
    n_layer=1, batch_size=3, eval_interval=1, eval_batches=1,
    precision=args.precision, device=args.device,
)
print("parameter dtype:", next(model.parameters()).dtype)
print("model device:", next(model.parameters()).device)
print("validation loss:", metrics["validation_after"])
assert next(model.parameters()).dtype == torch.float32
