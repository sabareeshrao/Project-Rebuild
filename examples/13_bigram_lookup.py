"""Lesson 13: a learnable table maps each current token to next-token scores."""

import torch

from rebuild_gpt.bigram import BigramLanguageModel

torch.manual_seed(0)
model = BigramLanguageModel(vocab_size=4)
ids = torch.tensor([[0, 1, 2], [3, 1, 0]])
logits = model(ids)

print("input shape:", tuple(ids.shape))
print("prediction shape:", tuple(logits.shape))  # (2, 3, 4)
print("parameters:", sum(p.numel() for p in model.parameters()))  # 4 x 4
print("same current token, same scores:", torch.equal(logits[0, 1], logits[1, 1]))
