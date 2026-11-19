"""Lesson 275: verify full training logits vs last-token inference logits."""

import torch

from rebuild_gpt.config import GPTConfig
from rebuild_gpt.gpt import GPTLanguageModel
from rebuild_gpt.model_invariants import check_forward_contract


def main() -> None:
    torch.manual_seed(17)
    model = GPTLanguageModel(GPTConfig(
        13, 5, 8, 2, 1, attention_impl="fused",
    ))
    x = torch.tensor([[0, 1, 2, 3], [5, 4, 3, 2]])
    y = torch.tensor([[1, 2, 3, -1], [4, 3, 2, -1]])
    print(check_forward_contract(model, x, y))


if __name__ == "__main__":
    main()
