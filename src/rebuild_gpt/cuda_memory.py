"""Measure CUDA allocator peaks during one operation.

This is a measurement of PyTorch-managed CUDA memory, not whole-device
VRAM usage as reported by a GPU driver.
"""

from collections.abc import Callable
from dataclasses import dataclass

import torch


@dataclass(frozen=True)
class CudaMemoryResult:
    baseline_allocated_bytes: int
    peak_allocated_bytes: int
    baseline_reserved_bytes: int
    peak_reserved_bytes: int

    @property
    def additional_peak_allocated_bytes(self) -> int:
        return max(0, self.peak_allocated_bytes - self.baseline_allocated_bytes)

    @property
    def additional_peak_reserved_bytes(self) -> int:
        return max(0, self.peak_reserved_bytes - self.baseline_reserved_bytes)


def measure_cuda_peak(
    operation: Callable[[], object], *, device: torch.device
) -> CudaMemoryResult:
    """Measure allocation peaks above a synchronized allocator baseline.

    Existing tensors (e.g. model weights) are part of the baseline. Warm
    up outside this function if you want kernel initialization excluded.
    """
    if not isinstance(device, torch.device) or device.type != "cuda":
        raise ValueError("CUDA memory profiling requires a cuda torch.device")
    if not torch.cuda.is_available():
        raise ValueError("CUDA memory profiling requires an available CUDA GPU")

    torch.cuda.synchronize(device)
    baseline_allocated = torch.cuda.memory_allocated(device)
    baseline_reserved = torch.cuda.memory_reserved(device)
    torch.cuda.reset_peak_memory_stats(device)
    operation()
    torch.cuda.synchronize(device)
    return CudaMemoryResult(
        baseline_allocated_bytes=baseline_allocated,
        peak_allocated_bytes=torch.cuda.max_memory_allocated(device),
        baseline_reserved_bytes=baseline_reserved,
        peak_reserved_bytes=torch.cuda.max_memory_reserved(device),
    )
