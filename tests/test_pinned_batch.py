"""Pinned host staging is an explicit GPU-only opt-in."""

import pytest
import torch

from rebuild_gpt.transfers import stage_cpu_batch


def test_unpinned_cpu_default_preserves_tensor_identity(monkeypatch):
    x = torch.arange(12).view(3, 4)
    y = torch.arange(12).view(3, 4)
    monkeypatch.setattr(
        torch.cuda, "is_available",
        lambda: (_ for _ in ()).throw(AssertionError("must not inspect CUDA")),
    )
    a, b = stage_cpu_batch(x, y, device=torch.device("cpu"))
    assert a is x and b is y


def test_pin_is_not_silently_accepted_on_cpu():
    x = torch.ones(2, 3)
    with pytest.raises(ValueError, match="requires a CUDA"):
        stage_cpu_batch(x, x, device=torch.device("cpu"), pin_memory=True)


def test_unavailable_cuda_fails_before_trying_to_pin(monkeypatch):
    x = torch.ones(2, 3)
    monkeypatch.setattr(torch.cuda, "is_available", lambda: False)
    with pytest.raises(ValueError, match="available CUDA"):
        stage_cpu_batch(x, x, device=torch.device("cuda"), pin_memory=True)


@pytest.mark.parametrize("bad", [0, 1, None, "true"])
def test_pin_flag_must_be_boolean(bad):
    x = torch.ones(1, 2)
    with pytest.raises(ValueError, match="boolean"):
        stage_cpu_batch(x, x, device=torch.device("cpu"), pin_memory=bad)


def test_inputs_must_be_tensors():
    with pytest.raises(ValueError, match="tensors"):
        stage_cpu_batch([1], torch.ones(1), device=torch.device("cpu"))


@pytest.mark.skipif(not torch.cuda.is_available(), reason="requires CUDA hardware")
def test_real_cuda_pinning_retains_values_and_is_idempotent():
    x = torch.arange(12, dtype=torch.long).view(3, 4)
    y = x + 1
    a, b = stage_cpu_batch(x, y, device=torch.device("cuda"), pin_memory=True)
    assert a.is_pinned() and b.is_pinned()
    assert a.device.type == b.device.type == "cpu"
    assert torch.equal(a, x) and torch.equal(b, y)
    same_a, same_b = stage_cpu_batch(
        a, b, device=torch.device("cuda"), pin_memory=True,
    )
    assert same_a is a and same_b is b
