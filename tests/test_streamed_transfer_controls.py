"""The streamed trainer stages once and transfers only current chunks."""

import pytest
import torch
from torch import nn

from rebuild_gpt.training import train_streamed_logical_batch_step


class TinyClassifier(nn.Module):
    def __init__(self):
        super().__init__()
        self.fc = nn.Linear(3, 4)

    def loss(self, x, y):
        return nn.functional.cross_entropy(self.fc(x), y)


def data():
    return torch.randn(5, 3), torch.tensor([0, 1, 2, 3, 0])


def test_default_stages_once_then_transfers_three_microbatches(monkeypatch):
    import rebuild_gpt.training as training

    staging = training.stage_cpu_batch
    transfer = training.transfer_microbatch
    staged = []
    transferred = []

    def spy_stage(x, y, **kwargs):
        staged.append((x.shape[0], kwargs["pin_memory"]))
        return staging(x, y, **kwargs)

    def spy_transfer(x, y, **kwargs):
        transferred.append((x.shape[0], kwargs["non_blocking"], kwargs["device"].type))
        return transfer(x, y, **kwargs)

    monkeypatch.setattr(training, "stage_cpu_batch", spy_stage)
    monkeypatch.setattr(training, "transfer_microbatch", spy_transfer)
    model = TinyClassifier()
    optimizer = torch.optim.AdamW(model.parameters(), lr=0.01)
    x, y = data()
    loss = train_streamed_logical_batch_step(
        model, optimizer, x, y, microbatch_size=2,
    )
    assert loss > 0
    assert staged == [(5, False)]
    assert transferred == [(2, False, "cpu"), (2, False, "cpu"), (1, False, "cpu")]


def test_cpu_pinning_is_explicit_error_not_silent_fallback():
    model = TinyClassifier()
    optimizer = torch.optim.AdamW(model.parameters())
    x, y = data()
    with pytest.raises(ValueError, match="pin_memory requires a CUDA"):
        train_streamed_logical_batch_step(
            model, optimizer, x, y, microbatch_size=2, pin_memory=True,
        )
    assert not optimizer.state


def test_nonblocking_requires_pinning_before_optimizer_update():
    model = TinyClassifier()
    optimizer = torch.optim.AdamW(model.parameters())
    x, y = data()
    with pytest.raises(ValueError, match="requires pin_memory"):
        train_streamed_logical_batch_step(
            model, optimizer, x, y, microbatch_size=2, non_blocking=True,
        )
    assert not optimizer.state


@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA unavailable")
def test_cuda_pinned_streaming_matches_blocking_training():
    import copy

    torch.manual_seed(148)
    regular = TinyClassifier().to("cuda")
    pinned = copy.deepcopy(regular)
    x, y = data()
    opt_regular = torch.optim.AdamW(regular.parameters(), lr=0.01)
    opt_pinned = torch.optim.AdamW(pinned.parameters(), lr=0.01)
    loss_regular = train_streamed_logical_batch_step(
        regular, opt_regular, x, y, microbatch_size=2,
    )
    loss_pinned = train_streamed_logical_batch_step(
        pinned, opt_pinned, x, y, microbatch_size=2,
        pin_memory=True, non_blocking=True,
    )
    assert loss_pinned == pytest.approx(loss_regular, abs=1e-5)
    for a, b in zip(regular.parameters(), pinned.parameters()):
        torch.testing.assert_close(a, b, atol=1e-5, rtol=1e-5)
