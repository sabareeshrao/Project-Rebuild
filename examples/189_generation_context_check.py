"""Lesson 296: verify long-prompt sampling with a cropped GPT context."""

import torch

from rebuild_gpt.config import GPTConfig
from rebuild_gpt.generation_context_check import check_generation_context
from rebuild_gpt.gpt import GPTLanguageModel


if __name__ == "__main__":
    torch.manual_seed(17)
    model = GPTLanguageModel(GPTConfig(
        11, 4, 8, 2, 1, attention_impl="fused",
    ))
    prompt = torch.tensor([[1, 2, 3, 4, 5, 6, 7]])
    print(check_generation_context(model, prompt, new_tokens=4))
