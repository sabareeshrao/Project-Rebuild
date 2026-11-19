"""Lesson 303: see why a constant offset changes neither softmax nor CE."""

import torch

from rebuild_gpt.logit_shift_invariance import check_logit_shift_invariance


if __name__ == "__main__":
    logits = torch.tensor([[[1., 0., -1.], [0., 2., 1.]]])
    targets = torch.tensor([[0, -1]])
    print(check_logit_shift_invariance(logits, targets, offset=3.25))
