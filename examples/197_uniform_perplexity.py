"""Lesson 304: calculate why uniform next-token guessing has perplexity V."""

import torch

from rebuild_gpt.perplexity_reference import compare_uniform_perplexity


if __name__ == "__main__":
    baseline_logits = torch.zeros(2, 3, 5)
    labels = torch.tensor([[0, 1, -1], [2, 3, 4]])
    print(compare_uniform_perplexity(baseline_logits, labels))
