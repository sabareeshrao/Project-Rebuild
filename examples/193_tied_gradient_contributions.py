"""Lesson 300: split tied GPT weight gradients into input and output paths."""

import torch

from rebuild_gpt.config import GPTConfig
from rebuild_gpt.gpt import GPTLanguageModel
from rebuild_gpt.tied_gradient_decomposition import check_tied_gradient_decomposition


if __name__ == "__main__":
    torch.manual_seed(17)
    model = GPTLanguageModel(GPTConfig(
        11, 5, 8, 2, 1, attention_impl="fused",
    ))
    x = torch.tensor([[0, 1, 2, 3], [4, 3, 2, 1]])
    y = torch.tensor([[1, 2, 3, 4], [3, 2, 1, 0]])
    print(check_tied_gradient_decomposition(model, x, y))
