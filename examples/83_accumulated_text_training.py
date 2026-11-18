"""Lesson 86: train with several microbatches per optimizer update."""

from rebuild_gpt.text_training import train_char_gpt

model, tokenizer, metrics = train_char_gpt(
    "data/tiny_story.txt",
    steps=12, batch_size=4, block_size=8,
    n_embd=16, n_head=4, n_layer=2,
    gradient_accumulation_steps=3,
    grouped_adamw=True, weight_decay=0.1,
    max_grad_norm=1.0,
)
print("optimizer update steps: 12")
print("microbatches per update: 3")
print("tokens per optimizer update:", int(metrics["effective_tokens_per_step"]))
print("final microbatch-averaged loss:", round(metrics["train_loss_last"], 4))
print("character vocabulary size:", tokenizer.vocab_size)
