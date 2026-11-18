"""Lesson 84: compare nanoGPT-style AdamW decay and no-decay parameters."""

from rebuild_gpt.config import GPTConfig
from rebuild_gpt.gpt import GPTLanguageModel
from rebuild_gpt.optimizers import configure_adamw

model = GPTLanguageModel(GPTConfig(
    vocab_size=20, block_size=8, n_embd=12, n_head=3,
    n_layer=2, attention_impl="fused",
))
optimizer = configure_adamw(
    model, learning_rate=0.001, weight_decay=0.1, betas=(0.9, 0.95),
)
for index, group in enumerate(optimizer.param_groups):
    print("group", index, "weight_decay", group["weight_decay"],
          "parameter tensors", len(group["params"]),
          "scalars", sum(p.numel() for p in group["params"]))
print("shared embedding registered once:",
      sum(model.tokens.lookup.weight is p for g in optimizer.param_groups for p in g["params"]) == 1)
