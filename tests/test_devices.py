import pytest
import torch

from rebuild_gpt.devices import resolve_device


def test_cpu_is_safe_default():
    assert resolve_device() == torch.device("cpu")
    assert resolve_device("cpu") == torch.device("cpu")


@pytest.mark.parametrize("invalid", ["mps", "gpu", "cuda:0", "", 42, None])
def test_unknown_device_is_rejected(invalid):
    with pytest.raises(ValueError, match="device"):
        resolve_device(invalid)


def test_auto_selects_available_backend(monkeypatch):
    monkeypatch.setattr(torch.cuda, "is_available", lambda: True)
    assert resolve_device("auto").type == "cuda"
    monkeypatch.setattr(torch.cuda, "is_available", lambda: False)
    assert resolve_device("auto").type == "cpu"


def test_explicit_cuda_never_silently_falls_back(monkeypatch):
    monkeypatch.setattr(torch.cuda, "is_available", lambda: False)
    with pytest.raises(ValueError, match="CUDA requested"):
        resolve_device("cuda")
