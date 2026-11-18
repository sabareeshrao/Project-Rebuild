import copy

import pytest
import torch
from torch import nn

from rebuild_gpt.training import train_accumulated_step, train_step


class TinyClassifier(nn.Module):
    def __init__(self):
        super().__init__()
        self.layer = nn.Linear(3, 4)

    def loss(self, inputs, targets):
        return nn.functional.cross_entropy(self.layer(inputs.float()), targets.long())


def samples():
    x = torch.tensor([[1., 0, 0], [0., 1, 0], [0., 0, 1], [1., 1, 1]])
    y = torch.tensor([1, 2, 3, 0])
    return [(x[:2], y[:2]), (x[2:], y[2:])]


def test_accumulation_matches_single_large_batch_update():
    torch.manual_seed(5)
    large = TinyClassifier()
    accumulated = copy.deepcopy(large)
    opts = [torch.optim.SGD(m.parameters(), lr=0.03) for m in (large, accumulated)]
    micro = samples()
    loss_full = train_step(large, opts[0], torch.cat([v[0] for v in micro]), torch.cat([v[1] for v in micro]))
    loss_accum = train_accumulated_step(accumulated, opts[1], micro)
    assert loss_accum == pytest.approx(loss_full, abs=1e-7)
    for p, q in zip(large.parameters(), accumulated.parameters()):
        torch.testing.assert_close(p, q, atol=1e-7, rtol=1e-6)


def test_accumulation_takes_one_optimizer_step_and_clips_once():
    torch.manual_seed(3)
    model = TinyClassifier()

    class CountingSGD(torch.optim.SGD):
        calls = 0

        def step(self, closure=None):
            self.calls += 1
            return super().step(closure=closure)

    optimizer = CountingSGD(model.parameters(), lr=0.1)
    loss = train_accumulated_step(model, optimizer, samples(), max_grad_norm=0.05)
    assert loss > 0
    assert optimizer.calls == 1
    assert torch.linalg.vector_norm(torch.cat([p.grad.flatten() for p in model.parameters()])) <= 0.05001


def test_accumulation_rejects_empty_or_differently_sized_batches():
    model = TinyClassifier()
    optimizer = torch.optim.SGD(model.parameters(), lr=0.1)
    with pytest.raises(ValueError, match="microbatches"):
        train_accumulated_step(model, optimizer, [])
    a, b = samples()
    with pytest.raises(ValueError, match="matching"):
        train_accumulated_step(model, optimizer, [a, (b[0][:1], b[1][:1])])


@pytest.mark.parametrize("bad", [0, -1, float("nan"), float("inf"), True])
def test_accumulation_rejects_bad_clipping_norm(bad):
    model = TinyClassifier()
    opt = torch.optim.SGD(model.parameters(), lr=0.1)
    with pytest.raises(ValueError, match="max_grad_norm"):
        train_accumulated_step(model, opt, samples(), max_grad_norm=bad)
