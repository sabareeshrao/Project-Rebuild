import pytest
import torch

from rebuild_gpt.benchmarks import TimingResult, time_operation


def test_summary_uses_median_min_max_not_the_last_sample():
    samples = TimingResult((5.0, 1.0, 4.0, 2.0))
    assert samples.median_ms == 3.0
    assert samples.min_ms == 1.0
    assert samples.max_ms == 5.0


def test_warmup_calls_are_not_counted_in_result():
    calls = []
    result = time_operation(
        lambda: calls.append(len(calls)),
        device=torch.device("cpu"), warmup=2, iterations=4,
    )
    assert len(calls) == 6
    assert len(result.samples_ms) == 4
    assert result.min_ms >= 0
    assert result.max_ms >= result.median_ms >= result.min_ms


@pytest.mark.parametrize("iterations,warmup", [(0, 0), (-1, 1), (1, -1), (True, 0), (1, False)])
def test_rejects_invalid_counts_before_running(iterations, warmup):
    calls = []
    with pytest.raises(ValueError):
        time_operation(
            lambda: calls.append(True), device=torch.device("cpu"),
            iterations=iterations, warmup=warmup,
        )
    assert not calls


def test_cuda_synchronizes_before_and_after_each_sample(monkeypatch):
    events = []
    import rebuild_gpt.benchmarks as b

    monkeypatch.setattr(
        torch.cuda, "synchronize", lambda device: events.append("sync")
    )
    monkeypatch.setattr(
        b, "perf_counter", iter([1.0, 1.002, 2.0, 2.004]).__next__
    )
    result = time_operation(
        lambda: events.append("call"),
        device=torch.device("cuda"), warmup=1, iterations=2,
    )
    assert result.samples_ms == pytest.approx((2.0, 4.0))
    assert events == [
        "call", "sync",
        "sync", "call", "sync",
        "sync", "call", "sync",
    ]


def test_only_cpu_and_cuda_are_supported():
    with pytest.raises(ValueError, match="device"):
        time_operation(lambda: None, device=torch.device("meta"))
