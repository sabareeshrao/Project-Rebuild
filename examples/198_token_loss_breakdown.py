"""Lesson 305: reconstruct masked GPT loss from individual token NLLs."""

import torch

from rebuild_gpt.token_loss_breakdown import inspect_token_losses


if __name__ == "__main__":
    x = torch.tensor([[[2., 0.], [4., -2.], [0., 5.]],
                      [[0., 3.], [1., -1.], [3., 0.]]])
    targets = torch.tensor([[0, -1, -1], [1, 0, -1]])
    print(inspect_token_losses(x, targets))
