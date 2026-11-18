"""Lesson 35: dropout changes attention output during training, not evaluation."""

import torch

from rebuild_gpt.multihead import MultiHeadAttention

torch.manual_seed(7)
attention = MultiHeadAttention(n_embd=8, n_head=2, dropout=1.0)
hidden = torch.randn(1, 3, 8)

attention.train()
print("training with dropout=1.0:", attention(hidden)[0, 0].tolist())
attention.eval()
print("evaluation, same weights:", attention(hidden)[0, 0].detach().tolist())
print("Use a moderate dropout such as 0.1 in real training, not 1.0.")
