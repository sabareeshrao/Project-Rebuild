"""Test autocast and the order of unscaling, clipping, and optimizer step."""

import torch
import pytest

from rebuild_gpt.precision import make_grad_scaler
from rebuild_gpt.training import (
    _finish_optimizer_step, train_accumulated_step, train_step,
)


class TinyLinear(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.net = torch.nn.Linear(4, 2)

    def loss(self, x, y):
        return torch.nn.functional.cross_entropy(self.net(x), y)


def batches():
    return [
        (torch.ones(3, 4), torch.tensor([0, 1, 0])),
        (torch.full((3, 4), 0.5), torch.tensor([1, 0, 1])),
    ]


def test_bf16_train_step_updates_fp32_weights():
    model = TinyLinear()
    opt = torch.optim.AdamW(model.parameters(), lr=0.01)
    old = model.net.weight.detach().clone()
    loss = train_step(model, opt, *batches()[0], precision="bf16")
    assert torch.isfinite(torch.tensor(loss))
    assert model.net.weight.dtype == torch.float32
    assert not torch.equal(old, model.net.weight)


def test_bf16_accumulation_trains_with_one_optimizer_update():
    model = TinyLinear()
    opt = torch.optim.AdamW(model.parameters(), lr=0.01)
    loss = train_accumulated_step(
        model, opt, batches(), precision="bf16", max_grad_norm=1.0
    )
    assert torch.isfinite(torch.tensor(loss))
    assert len(opt.state) > 0
    assert all(int(state["step"].item()) == 1 for state in opt.state.values())


def test_fp16_requires_a_scaler_and_cuda():
    model = TinyLinear()
    opt = torch.optim.AdamW(model.parameters())
    with pytest.raises(ValueError, match="requires CUDA"):
        train_step(model, opt, *batches()[0], precision="fp16")
    assert make_grad_scaler(torch.device("cpu"), "bf16") is None


def test_gradients_are_unscaled_before_clipping():
    model = torch.nn.Linear(1, 1, bias=False)
    opt = torch.optim.SGD(model.parameters(), lr=0.1)
    model.weight.grad = torch.tensor([[8.0]])
    calls = []

    class TestScaler:
        def unscale_(self, optimizer):
            calls.append("unscale")
            model.weight.grad.div_(4)

        def step(self, optimizer):
            calls.append("step")
            optimizer.step()

        def update(self):
            calls.append("update")

    _finish_optimizer_step(model, opt, 1.0, TestScaler())
    assert calls == ["unscale", "step", "update"]
    assert model.weight.grad.abs().item() <= 1.001
