"""Lesson 281: compare full GPT logits and gradients using manual vs SDPA."""

import torch

from rebuild_gpt.attention_backend_parity import compare_attention_backends
from rebuild_gpt.config import GPTConfig
from rebuild_gpt.gpt import GPTLanguageModel


def main() -> None:
    torch.manual_seed(17)
    model = GPTLanguageModel(GPTConfig(
        11, 5, 8, 2, 1, attention_impl="fused",
    ))
    x = torch.tensor([[0, 1, 2, 3]])
    y = torch.tensor([[1, 2, 3, -1]])
    print(compare_attention_backends(model, x, y))


if __name__ == "__main__":
    main()
