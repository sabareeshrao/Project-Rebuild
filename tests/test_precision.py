import pytest
import torch

from rebuild_gpt.precision import autocast_context, resolve_precision


def test_fp32_stays_at_full_precision_on_cpu():
    assert resolve_precision(torch.device("cpu"), "fp32") is None
    with autocast_context(torch.device("cpu"), "fp32"):
        assert torch.mm(torch.ones(2, 2), torch.ones(2, 2)).dtype == torch.float32


def test_bf16_autocasts_cpu_matrix_multiplication():
    assert resolve_precision(torch.device("cpu"), "bf16") == torch.bfloat16
    with autocast_context(torch.device("cpu"), "bf16"):
        result = torch.mm(torch.ones(2, 2), torch.ones(2, 2))
    assert result.dtype == torch.bfloat16


@pytest.mark.parametrize("precision", ["auto", "float16", "", None])
def test_precision_rejects_unknown_name(precision):
    with pytest.raises(ValueError, match="precision"):
        resolve_precision(torch.device("cpu"), precision)


def test_cpu_fp16_is_rejected_instead_of_silently_falling_back():
    with pytest.raises(ValueError, match="requires CUDA"):
        resolve_precision(torch.device("cpu"), "fp16")


def test_cuda_bf16_checks_device_capability(monkeypatch):
    monkeypatch.setattr(torch.cuda, "is_available", lambda: True)
    monkeypatch.setattr(torch.cuda, "is_bf16_supported", lambda: False)
    with pytest.raises(ValueError, match="does not support bf16"):
        resolve_precision(torch.device("cuda"), "bf16")
