"""Lesson 81: opt in to gradient clipping and cosine warmup-decay."""

from rebuild_gpt.text_training import train_char_gpt

history = []
model, tokenizer, metrics = train_char_gpt(
    "data/tiny_story.txt",
    steps=30, eval_interval=10, eval_batches=3, history=history,
    max_grad_norm=1.0,
    warmup_steps=3, lr_decay_steps=25, min_lr=0.002,
)
for entry in history:
    print(f"step {entry['step']:>2}: val={entry['validation_loss']:.4f}")
print("last optimizer learning rate:", round(metrics["learning_rate_last"], 6))
print("vocab size:", tokenizer.vocab_size)
