"""Lesson 79: inspect nanoGPT's linear warmup and cosine learning-rate decay."""

from rebuild_gpt.schedule import cosine_warmup_lr

for step in range(9):
    rate = cosine_warmup_lr(
        step, max_lr=0.01, min_lr=0.001,
        warmup_steps=2, decay_steps=6,
    )
    print(f"iteration={step}: learning rate={rate:.6f}")
