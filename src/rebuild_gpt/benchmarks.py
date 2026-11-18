"""Small, dependency-free timing primitives for reproducible experiments."""

from collections.abc import Callable
from dataclasses import dataclass
from statistics import median
from time import perf_counter

import torch


@dataclass(frozen=True)
class TimingResult:
    """Summary of wall-clock samples measured after warmup (milliseconds)."""

    samples_ms: tuple[float, ...]

    @property
    def median_ms(self) -> float:
        return median(self.samples_ms)

    @property
    def min_ms(self) -> float:
        return min(self.samples_ms)

    @property
    def max_ms(self) -> float:
        return max(self.samples_ms)


def time_operation(
    operation: Callable[[], object],
    *,
    device: torch.device,
    iterations: int = 10,
    warmup: int = 3,
) -> TimingResult:
    """Time repeated calls; synchronize asynchronous CUDA execution correctly.

    Only use comparable operations with the same inputs and warmup conditions.
    This is a microbenchmark, not a hardware-independent performance claim.
    """
    if type(iterations) is not int or iterations < 1:
        raise ValueError("iterations must be a positive integer")
    if type(warmup) is not int or warmup < 0:
        raise ValueError("warmup must be a nonnegative integer")
    if not isinstance(device, torch.device) or device.type not in ("cpu", "cuda"):
        raise ValueError("device must be a CPU or CUDA torch.device")

    def sync() -> None:
        if device.type == "cuda":
            torch.cuda.synchronize(device)

    for _ in range(warmup):
        operation()
    sync()

    samples = []
    for _ in range(iterations):
        sync()
        began = perf_counter()
        operation()
        sync()
        samples.append((perf_counter() - began) * 1000)
    return TimingResult(tuple(samples))
