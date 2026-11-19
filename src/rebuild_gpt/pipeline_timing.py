"""Time each sequential training pipeline stage with CUDA synchronization.

Instrumentation prevents transfer/compute overlap. Use this educational
breakdown to find candidates for profiling, not as an overlap benchmark.
"""

from collections.abc import Callable
from dataclasses import dataclass
from time import perf_counter

import torch

from rebuild_gpt.benchmarks import TimingResult


@dataclass(frozen=True)
class PipelineTiming:
    sampling: TimingResult
    transfer: TimingResult
    update: TimingResult
    end_to_end: TimingResult

    @property
    def slowest_stage(self) -> str:
        """Name the stage with the highest median (ties in pipeline order)."""
        medians = (
            ("sampling", self.sampling.median_ms),
            ("transfer", self.transfer.median_ms),
            ("update", self.update.median_ms),
        )
        return max(medians, key=lambda pair: pair[1])[0]


def time_training_pipeline(
    sample: Callable[[], object],
    transfer: Callable[[object], object],
    update: Callable[[object], object],
    *,
    device: torch.device,
    warmup: int = 2,
    iterations: int = 5,
) -> PipelineTiming:
    """Time serial sample -> transfer -> optimizer update, in milliseconds.

    A CUDA sync follows EACH stage, ensuring asynchronous launches are
    attributed to their originating stage. This intentionally serializes
    all work; it cannot measure pipelined overlap.
    """
    if not isinstance(device, torch.device) or device.type not in ("cpu", "cuda"):
        raise ValueError("device must be a CPU or CUDA torch.device")
    if type(iterations) is not int or iterations < 1:
        raise ValueError("iterations must be a positive integer")
    if type(warmup) is not int or warmup < 0:
        raise ValueError("warmup must be a nonnegative integer")

    def sync() -> None:
        if device.type == "cuda":
            torch.cuda.synchronize(device)

    samples: list[float] = []
    transfers: list[float] = []
    updates: list[float] = []
    totals: list[float] = []
    for index in range(warmup + iterations):
        sync()
        beginning = perf_counter()
        batch = sample()
        sync()
        after_sample = perf_counter()
        on_device = transfer(batch)
        sync()
        after_transfer = perf_counter()
        update(on_device)
        sync()
        after_update = perf_counter()
        if index >= warmup:
            samples.append((after_sample - beginning) * 1000.0)
            transfers.append((after_transfer - after_sample) * 1000.0)
            updates.append((after_update - after_transfer) * 1000.0)
            totals.append((after_update - beginning) * 1000.0)
    return PipelineTiming(
        sampling=TimingResult(tuple(samples)),
        transfer=TimingResult(tuple(transfers)),
        update=TimingResult(tuple(updates)),
        end_to_end=TimingResult(tuple(totals)),
    )
