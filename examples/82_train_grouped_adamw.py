"""Lesson 85: use nanoGPT-style AdamW parameter grouping in text training."""

from rebuild_gpt.text_training import train_char_gpt

options = dict(
    steps=10, block_size=8, n_embd=16, n_head=4, n_layer=2,
    grouped_adamw=True, weight_decay=0.1, adam_betas=(0.9, 0.95),
)
model, tokenizer, metrics = train_char_gpt("data/tiny_story.txt", **options)
print("two optimizer groups:", metrics["optimizer_group_count"])
print("final training-batch loss:", round(metrics["train_loss_last"], 4))
print("vocabulary size:", tokenizer.vocab_size)
print("previous tutorials still use their original AdamW settings by default")
