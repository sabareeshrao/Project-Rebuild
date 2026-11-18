import pytest
import torch

from rebuild_gpt.cuda_memory import CudaMemoryResult, measure_cuda_peak


def test_peak_deltas_do_not_confuse_baseline_with_new_allocation():
    result = CudaMemoryResult(
        baseline_allocated_bytes=400,
        peak_allocated_bytes=950,
        baseline_reserved_bytes=1024,
        peak_reserved_bytes=2048,
    )
    assert result.additional_peak_allocated_bytes == 550
    assert result.additional_peak_reserved_bytes == 1024


def test_cpu_is_rejected_before_touching_cuda(monkeypatch):
    monkeypatch.setattr(torch.cuda, "is_available", lambda: (_ for _ in ()).throw(
        AssertionError("CUDA should not be queried")
    ))
    with pytest.raises(ValueError, match="cuda torch.device"):
        measure_cuda_peak(lambda: None, device=torch.device("cpu"))


def test_unavailable_cuda_is_rejected(monkeypatch):
    monkeypatch.setattr(torch.cuda, "is_available", lambda: False)
    with pytest.raises(ValueError, match="available CUDA GPU"):
        measure_cuda_peak(lambda: None, device=torch.device("cuda"))


def test_measurement_synchronizes_and_resets_peak_after_baseline(monkeypatch):
    calls = []
    monkeypatch.setattr(torch.cuda, "is_available", lambda: True)
    monkeypatch.setattr(torch.cuda, "synchronize", lambda d: calls.append("sync"))
    monkeypatch.setattr(torch.cuda, "memory_allocated",
                        lambda d: calls.append("allocated") or 128)
    monkeypatch.setattr(torch.cuda, "memory_reserved",
                        lambda d: calls.append("reserved") or 256)
    monkeypatch.setattr(torch.cuda, "reset_peak_memory_stats",
                        lambda d: calls.append("reset"))
    monkeypatch.setattr(torch.cuda, "max_memory_allocated",
                        lambda d: calls.append("peak_allocated") or 320)
    monkeypatch.setattr(torch.cuda, "max_memory_reserved",
                        lambda d: calls.append("peak_reserved") or 512)
    result = measure_cuda_peak(
        lambda: calls.append("operation"), device=torch.device("cuda")
    )
    assert calls == [
        "sync", "allocated", "reserved", "reset", "operation",
        "sync", "peak_allocated", "peak_reserved",
    ]
    assert result.additional_peak_allocated_bytes == 192


@pytest.mark.skipif(not torch.cuda.is_available(), reason="requires CUDA GPU")
def test_real_cuda_allocation_has_peak_above_baseline():
    device = torch.device("cuda")
    result = measure_cuda_peak(
        lambda: torch.ones(100_000, device=device) * 2, device=device
    )
    assert result.peak_allocated_bytes >= result.baseline_allocated_bytes
    assert result.peak_reserved_bytes >= result.baseline_reserved_bytes
