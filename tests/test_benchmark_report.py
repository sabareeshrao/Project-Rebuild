import json

import pytest
import torch

from rebuild_gpt.benchmark_report import build_benchmark_report, write_benchmark_report


def test_cpu_metadata_does_not_probe_cuda(monkeypatch, tmp_path):
    def unexpected(*args, **kwargs):
        raise AssertionError("CUDA should not be queried on CPU")
    monkeypatch.setattr(torch.cuda, "get_device_name", unexpected)
    settings = {"seed": 134, "batch_sizes": [1, 2]}
    rows = [{"batch_size": 1, "median_step_ms": 1.25}]
    report = build_benchmark_report(
        rows=rows, device=torch.device("cpu"), settings=settings,
    )
    assert report["schema_version"] == 1
    assert report["environment"]["device"] == "cpu"
    assert report["environment"]["cuda_device_name"] is None
    assert report["environment"]["pytorch"]
    assert report["environment"]["python"]
    assert report["settings"] == settings
    path = tmp_path / "nested" / "report.json"
    write_benchmark_report(path, report)
    assert json.loads(path.read_text(encoding="utf-8")) == report


def test_report_is_independent_of_original_mutable_inputs():
    rows = [{"batch_size": 1}]
    settings = {"seed": 1}
    report = build_benchmark_report(
        rows=rows, settings=settings, device=torch.device("cpu"),
    )
    rows[0]["batch_size"] = 2
    settings["seed"] = 7
    assert report["results"][0]["batch_size"] == 1
    assert report["settings"]["seed"] == 1


def test_invalid_report_arguments_are_rejected():
    with pytest.raises(ValueError, match="rows"):
        build_benchmark_report(rows=[], device=torch.device("cpu"), settings={})
    with pytest.raises(ValueError, match="device"):
        build_benchmark_report(rows=[{}], device=torch.device("meta"), settings={})
    with pytest.raises(ValueError, match="settings"):
        build_benchmark_report(rows=[{}], device=torch.device("cpu"), settings=None)
