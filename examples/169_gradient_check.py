"""Lesson 276: independently measure one tiny GPT weight's loss slope."""

import torch

from rebuild_gpt.config import GPTConfig
from rebuild_gpt.gpt import GPTLanguageModel
from rebuild_gpt.gradient_check import check_parameter_gradient


def main() -> None:
    torch.manual_seed(31)
    model = GPTLanguageModel(GPTConfig(
        9, 5, 8, 2, 1, attention_impl="fused",
    )).double()
    x = torch.tensor([[0, 1, 2, 3]])
    y = torch.tensor([[1, 2, 3, -1]])
    print(check_parameter_gradient(model, x, y))


if __name__ == "__main__":
    main()
