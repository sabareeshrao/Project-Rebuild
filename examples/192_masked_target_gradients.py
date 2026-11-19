"""Lesson 299: prove target -1 is ignored in both GPT loss and gradients."""

import torch

from rebuild_gpt.config import GPTConfig
from rebuild_gpt.gpt import GPTLanguageModel
from rebuild_gpt.masked_gradient_parity import check_masked_target_gradients


if __name__ == "__main__":
    torch.manual_seed(17)
    model = GPTLanguageModel(GPTConfig(
        11, 5, 8, 2, 1, attention_impl="fused",
    ))
    x = torch.tensor([[0, 1, 2, 3], [3, 2, 1, 0]])
    y = torch.tensor([[1, -1, 3, -1], [2, 1, -1, 1]])
    print(check_masked_target_gradients(model, x, y))
