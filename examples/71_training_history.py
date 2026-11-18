"""Lesson 74: print periodic loss estimates for a tiny character GPT."""

from rebuild_gpt.text_training import train_char_gpt

history: list[dict[str, float | int]] = []
_, _, metrics = train_char_gpt(
    "data/tiny_story.txt", steps=30, eval_interval=10,
    eval_batches=3, history=history,
)
for entry in history:
    print(
        f"step={entry['step']:>3}  train={entry['train_loss']:.4f} "
        f"validation={entry['validation_loss']:.4f}"
    )
print("last training batch loss:", round(metrics["train_loss_last"], 4))
