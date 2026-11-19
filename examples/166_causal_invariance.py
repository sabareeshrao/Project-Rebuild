"""Lesson 273: test future-token isolation without comparing to a target logit."""

import torch

from rebuild_gpt.config import GPTConfig
from rebuild_gpt.gpt import GPTLanguageModel
from rebuild_gpt.model_invariants import check_causal_prefix


def main() -> None:
    torch.manual_seed(17)
    config = GPTConfig(11, 6, 12, 3, 2, attention_impl="fused")
    model = GPTLanguageModel(config)
    tokens = torch.tensor([[1, 2, 3, 4], [4, 3, 2, 1]])
    print(check_causal_prefix(model, tokens))


if __name__ == "__main__":
    main()
