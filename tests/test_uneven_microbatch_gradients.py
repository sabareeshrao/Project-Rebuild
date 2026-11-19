"""Uneven chunks must equal a one-shot logical batch update."""

import copy

import pytest
import torch
from torch import nn

from rebuild_gpt.microbatches import split_microbatches
from rebuild_gpt.training import train_accumulated_step, train_step


class TinyClassifier(nn.Module):
    def __init__(self):
        super().__init__()
        self.linear = nn.Linear(3, 4)

    def loss(self, inputs, targets):
        return nn.functional.cross_entropy(self.linear(inputs), targets)


def examples():
    x = torch.tensor([
        [0.2, 1., 0.], [0., 1., 0.5], [0., 0., 1.],
        [1., 1., 0.5], [2., 0., 0.],
    ])
    y = torch.tensor([1, 2, 3, 0, 1])
    return x, y


@pytest.mark.parametrize("optimizer_class", [torch.optim.SGD, torch.optim.AdamW])
@pytest.mark.parametrize("microbatch_size", [1, 2, 3, 4, 5])
def test_uneven_microbatch_matches_full_update(optimizer_class, microbatch_size):
    torch.manual_seed(137)
    full = TinyClassifier()
    split = copy.deepcopy(full)
    x, y = examples()
    optimizer_full = optimizer_class(full.parameters(), lr=0.01)
    optimizer_split = optimizer_class(split.parameters(), lr=0.01)
    full_loss = train_step(full, optimizer_full, x, y)
    chunks = split_microbatches(x, y, microbatch_size=microbatch_size)
    split_loss = train_accumulated_step(split, optimizer_split, chunks)
    assert split_loss == pytest.approx(full_loss, abs=2e-6)
    for a, b in zip(full.parameters(), split.parameters()):
        torch.testing.assert_close(a, b, atol=2e-5, rtol=2e-5)


def test_uneven_clipped_accumulation_steps_once():
    model = TinyClassifier()
    x, y = examples()
    class CountingSGD(torch.optim.SGD):
        calls = 0
        def step(self, closure=None):
            self.calls += 1
            return super().step(closure=closure)

    optimizer = CountingSGD(model.parameters(), lr=0.02)
    loss = train_accumulated_step(
        model, optimizer, split_microbatches(x, y, microbatch_size=2),
        max_grad_norm=0.04,
    )
    assert loss > 0
    assert optimizer.calls == 1
    gradient_norm = torch.linalg.vector_norm(
        torch.cat([p.grad.flatten() for p in model.parameters()])
    )
    assert gradient_norm <= 0.04001


def test_incompatible_feature_shapes_are_still_rejected():
    x, y = examples()
    model = TinyClassifier()
    optimizer = torch.optim.AdamW(model.parameters())
    with pytest.raises(ValueError, match="matching"):
        train_accumulated_step(
            model, optimizer, [(x[:2], y[:2]), (torch.zeros(3, 5), y[2:])]
        )
