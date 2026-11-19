"""Lesson 306: see why uneven masked microbatches need weighted CE."""

from rebuild_gpt.masked_microbatch_loss import check_masked_microbatch_equivalence

import torch


if __name__ == "__main__":
    scores = torch.tensor([[[4., 0.], [1., 2.], [2., 1.]],
                           [[-3., 5.], [3., -1.], [5., -2.]]])
    labels = torch.tensor([[0, -1, -1], [0, 1, 1]])
    print(check_masked_microbatch_equivalence(scores, labels))
