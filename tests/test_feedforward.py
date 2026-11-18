import pytest
import torch

from rebuild_gpt.feedforward import FeedForward


def test_feedforward_expands_and_restores_width():
    torch.manual_seed(2)
    mlp = FeedForward(n_embd=6)
    x = torch.randn(2, 4, 6, requires_grad=True)
    y = mlp(x)
    assert mlp.expand.weight.shape == (24, 6)
    assert mlp.project.weight.shape == (6, 24)
    assert y.shape == x.shape
    y.square().mean().backward()
    assert x.grad is not None
    assert all(p.grad is not None for p in mlp.parameters())


def test_feedforward_never_mixes_time_positions():
    mlp = FeedForward(n_embd=4)
    x = torch.randn(2, 4, 4)
    changed = x.clone()
    changed[:, -1] += 100
    assert torch.allclose(mlp(x)[:, :-1], mlp(changed)[:, :-1], atol=1e-6)


def test_feedforward_dropout_only_during_training():
    mlp = FeedForward(n_embd=4, dropout=1.0)
    x = torch.randn(1, 2, 4)
    mlp.train()
    assert torch.count_nonzero(mlp(x)) == 0
    mlp.eval()
    assert torch.count_nonzero(mlp(x)) > 0


@pytest.mark.parametrize("rate", [-0.1, 1.01, float("nan")])
def test_feedforward_rejects_invalid_dropout(rate):
    with pytest.raises(ValueError, match="dropout"):
        FeedForward(4, dropout=rate)
