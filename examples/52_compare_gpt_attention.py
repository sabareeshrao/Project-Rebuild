"""Lesson 52: switch the stacked GPT between readable and fused attention."""

import torch
from rebuild_gpt.config import GPTConfig
from rebuild_gpt.gpt import GPTLanguageModel
from rebuild_gpt.parameters import count_parameters

torch.manual_seed(7)
for impl in ("educational", "fused"):
    config = GPTConfig(vocab_size=12, block_size=8, n_embd=16,
                       n_head=4, n_layer=2, attention_impl=impl)
    model = GPTLanguageModel(config)
    tokens = torch.tensor([[0, 1, 2, 3]])
    print(impl, "logits shape:", tuple(model(tokens).shape),
          "parameters:", count_parameters(model))
    print("  first attention:", type(model.blocks.blocks[0].attention).__name__)
print("Different random weights do NOT imply identical output logits.")
