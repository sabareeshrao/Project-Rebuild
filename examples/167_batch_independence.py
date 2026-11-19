"""Lesson 274: batch rows should behave like separate GPT examples."""

import torch

from rebuild_gpt.config import GPTConfig
from rebuild_gpt.gpt import GPTLanguageModel
from rebuild_gpt.model_invariants import check_batch_independence


def main() -> None:
    torch.manual_seed(17)
    model = GPTLanguageModel(GPTConfig(
        13, 6, 12, 3, 2, attention_impl="sdpa",
    ))
    inputs = torch.tensor([[0, 1, 2, 3], [9, 8, 7, 6], [2, 4, 6, 8]])
    print(check_batch_independence(model, inputs))


if __name__ == "__main__":
    main()
