"""Lesson 279: sample a GPT twice with matching private RNG seeds."""

import torch

from rebuild_gpt.config import GPTConfig
from rebuild_gpt.gpt import GPTLanguageModel
from rebuild_gpt.seeded_generation_check import check_seeded_generation


def main() -> None:
    torch.manual_seed(17)
    model = GPTLanguageModel(GPTConfig(11, 5, 8, 2, 1, attention_impl="fused"))
    print(check_seeded_generation(model, torch.tensor([[1, 2, 3]]), seed=41))


if __name__ == "__main__":
    main()
