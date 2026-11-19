"""Streaming inputs is numerically equivalent to a full batch optimizer step."""

import copy

import pytest
import torch
from torch import nn

from rebuild_gpt.training import (
    train_logical_batch_step, train_step, train_streamed_logical_batch_step,
)


class Classifier(nn.Module):
    def __init__(self):
        super().__init__()
        self.linear = nn.Linear(3, 4)

    def loss(self, x, y):
        assert x.device == self.linear.weight.device
        assert y.device == self.linear.weight.device
        return nn.functional.cross_entropy(self.linear(x), y)


def data():
    return (
        torch.tensor([
            [1., 0, 0], [0., 1, 0], [0., 0, 1],
            [1., 1, 0], [0.4, 0.6, 0], [1., 0, 1],
            [0.8, 0, 0.2],
        ]),
        torch.tensor([1, 2, 3, 0, 2, 1, 0]),
    )


@pytest.mark.parametrize("size", [1, 2, 3, 4, 7])
@pytest.mark.parametrize("optimizer_class", [torch.optim.AdamW, torch.optim.SGD])
def test_streamed_matches_full_and_existing_accumulator(size, optimizer_class):
    torch.manual_seed(142)
    full = Classifier()
    streamed = copy.deepcopy(full)
    buffered = copy.deepcopy(full)
    x, y = data()
    options = [optimizer_class(m.parameters(), lr=0.01) for m in (full, streamed, buffered)]
    loss_full = train_step(full, options[0], x, y)
    loss_streamed = train_streamed_logical_batch_step(
        streamed, options[1], x, y, microbatch_size=size,
    )
    loss_buffered = train_logical_batch_step(
        buffered, options[2], x, y, microbatch_size=size,
    )
    assert loss_streamed == pytest.approx(loss_full, abs=2e-6)
    assert loss_streamed == pytest.approx(loss_buffered, abs=2e-6)
    for a, b, c in zip(full.parameters(), streamed.parameters(), buffered.parameters()):
        torch.testing.assert_close(a, b, atol=2e-5, rtol=2e-5)
        torch.testing.assert_close(b, c, atol=2e-5, rtol=2e-5)


def test_gradient_clipping_and_one_step_are_preserved():
    class CountingSGD(torch.optim.SGD):
        calls = 0

        def step(self, closure=None):
            self.calls += 1
            return super().step(closure=closure)

    model = Classifier()
    optimizer = CountingSGD(model.parameters(), lr=0.01)
    x, y = data()
    train_streamed_logical_batch_step(
        model, optimizer, x, y, microbatch_size=3, max_grad_norm=0.05,
    )
    assert optimizer.calls == 1
    norm = torch.linalg.vector_norm(
        torch.cat([p.grad.reshape(-1) for p in model.parameters()])
    )
    assert norm <= 0.05001


def test_invalid_microbatch_fails_before_updating_optimizer():
    model = Classifier()
    optimizer = torch.optim.AdamW(model.parameters())
    x, y = data()
    with pytest.raises(ValueError, match="microbatch_size"):
        train_streamed_logical_batch_step(
            model, optimizer, x, y, microbatch_size=0,
        )
    assert not optimizer.state


def test_cpu_bf16_streaming_preserves_fp32_parameters():
    model = Classifier()
    optimizer = torch.optim.AdamW(model.parameters(), lr=0.01)
    x, y = data()
    loss = train_streamed_logical_batch_step(
        model, optimizer, x, y, microbatch_size=3, precision="bf16",
    )
    assert torch.isfinite(torch.tensor(loss))
    assert all(p.dtype == torch.float32 for p in model.parameters())


@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA unavailable")
def test_cpu_logical_batch_streams_into_cuda_model():
    model = Classifier().to("cuda")
    optimizer = torch.optim.AdamW(model.parameters(), lr=0.01)
    x, y = data()
    loss = train_streamed_logical_batch_step(
        model, optimizer, x, y, microbatch_size=2,
    )
    assert loss > 0
    assert all(p.device.type == "cuda" for p in model.parameters())
