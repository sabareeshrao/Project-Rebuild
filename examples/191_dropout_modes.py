"""Lesson 298: compare GPT dropout in training and evaluation modes."""

import torch

from rebuild_gpt.config import GPTConfig
from rebuild_gpt.dropout_semantics import check_dropout_semantics
from rebuild_gpt.gpt import GPTLanguageModel


if __name__ == "__main__":
    torch.manual_seed(17)
    model = GPTLanguageModel(GPTConfig(
        11, 5, 12, 3, 2, dropout=0.4, attention_impl="fused",
    ))
    x = torch.tensor([[1, 2, 3, 4], [4, 3, 2, 1]])
    print(check_dropout_semantics(model, x))
