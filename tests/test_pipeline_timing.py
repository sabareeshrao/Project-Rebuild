"""Stage timings should execute in order and synchronize at boundaries."""

import pytest
import torch

from rebuild_gpt.pipeline_timing import time_training_pipeline


def test_stage_order_and_warmup_are_explicit():
    events = []

    def sample():
        events.append("sample")
        return 3

    def transfer(x):
        events.append("transfer")
        return x * 2

    def update(x):
        events.append(f"update:{x}")

    result = time_training_pipeline(
        sample, transfer, update,
        device=torch.device("cpu"), warmup=2, iterations=3,
    )
    assert events == ["sample", "transfer", "update:6"] * 5
    assert len(result.end_to_end.samples_ms) == 3
    assert result.end_to_end.median_ms >= result.sampling.min_ms
    assert result.slowest_stage in ("sampling", "transfer", "update")


def test_cuda_synchronization_occurs_after_each_stage(monkeypatch):
    calls = []
    monkeypatch.setattr(
        torch.cuda, "synchronize", lambda device: calls.append("sync")
    )
    result = time_training_pipeline(
        lambda: calls.append("sample") or 4,
        lambda x: calls.append("transfer") or x,
        lambda x: calls.append("update"),
        device=torch.device("cuda"), warmup=0, iterations=1,
    )
    assert calls == [
        "sync", "sample", "sync", "transfer",
        "sync", "update", "sync",
    ]
    assert len(result.sampling.samples_ms) == 1


def test_stage_durations_sum_to_total_for_each_iteration():
    result = time_training_pipeline(
        lambda: 1,
        lambda x: x + 2,
        lambda x: x,
        device=torch.device("cpu"), warmup=0, iterations=3,
    )
    for sample, transfer, update, total in zip(
        result.sampling.samples_ms, result.transfer.samples_ms,
        result.update.samples_ms, result.end_to_end.samples_ms,
    ):
        assert total == pytest.approx(sample + transfer + update)


@pytest.mark.parametrize("warmup,iterations", [(0, 0), (-1, 1), (1, -1), (True, 1)])
def test_invalid_counts_fail_before_work(warmup, iterations):
    events = []
    with pytest.raises(ValueError):
        time_training_pipeline(
            lambda: events.append("sample"), lambda x: x, lambda x: None,
            device=torch.device("cpu"), warmup=warmup, iterations=iterations,
        )
    assert events == []
