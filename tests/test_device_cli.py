"""Device selection is explicit and preserves CPU training behavior."""

import json

import pytest
import torch

from rebuild_gpt.train_cli import main


def args(tmp_path):
    corpus = tmp_path / "text.txt"
    corpus.write_text("ab" * 90, encoding="utf-8")
    return [
        "--corpus", str(corpus), "--steps", "2",
        "--block-size", "3", "--n-embd", "8", "--n-head", "2",
        "--n-layer", "1", "--batch-size", "3",
        "--eval-every", "1", "--eval-batches", "1",
        "--output", str(tmp_path / "model.pt"),
        "--best-output", str(tmp_path / "best.pt"),
        "--history-file", str(tmp_path / "history.json"),
    ]


def test_cpu_device_cli_records_requested_device(tmp_path):
    assert main(args(tmp_path) + ["--device", "cpu"]) == 0
    history = json.loads((tmp_path / "history.json").read_text())
    assert history["requested_device"] == "cpu"
    assert history["metrics"]["validation_after"] > 0


def test_cpu_auto_fallback_and_unsupported_device(tmp_path, monkeypatch):
    monkeypatch.setattr(torch.cuda, "is_available", lambda: False)
    assert main(args(tmp_path) + ["--device", "auto"]) == 0
    with pytest.raises(SystemExit) as exc:
        main(args(tmp_path) + ["--device", "cuda"])
    assert exc.value.code == 2
    with pytest.raises(SystemExit) as exc:
        main(args(tmp_path) + ["--device", "mps"])
    assert exc.value.code == 2


@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA unavailable")
def test_cuda_device_cli_writes_portable_inference_checkpoint(tmp_path):
    from rebuild_gpt.text_checkpoint import load_text_checkpoint

    assert main(args(tmp_path) + ["--device", "cuda"]) == 0
    model, _ = load_text_checkpoint(tmp_path / "model.pt")
    assert next(model.parameters()).device.type == "cpu"
