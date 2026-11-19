"""Lesson 301: inspect the exact next-token sampling probabilities."""

import torch

from rebuild_gpt.config import GPTConfig
from rebuild_gpt.gpt import GPTLanguageModel
from rebuild_gpt.next_token_distribution import inspect_next_token_distribution


if __name__ == "__main__":
    torch.manual_seed(17)
    model = GPTLanguageModel(GPTConfig(
        7, 4, 8, 2, 1, attention_impl="fused",
    ))
    report = inspect_next_token_distribution(
        model, torch.tensor([[0, 1, 2, 3, 4]]),
        temperature=0.8, top_k=3,
    )
    print("candidate IDs:", report["allowed_ids"])
    print("entropy (nats):", report["entropy_nats"])
    print("full probabilities:", report["probabilities"])
