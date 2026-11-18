"""Lesson 31: predict next-token logits using earlier token context."""

import torch

from rebuild_gpt.single_head_model import SingleHeadLanguageModel

torch.manual_seed(3)
model = SingleHeadLanguageModel(vocab_size=4, block_size=6, n_embd=8, head_size=4)
tokens = torch.tensor([[0, 1, 2], [3, 1, 2]])
targets = torch.tensor([[1, 2, 3], [1, 2, 0]])
logits = model(tokens)
loss = model.loss(tokens, targets)

print("input tokens [B,T]:", tuple(tokens.shape))
print("context-aware logits [B,T,V]:", tuple(logits.shape))
print("cross-entropy loss:", round(loss.item(), 4))
print("The attention head now mixes earlier token information.")
