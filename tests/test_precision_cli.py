"""CPU BF16 CLI, checkpoint metadata and error handling."""

import json

import pytest
import torch

from rebuild_gpt.train_cli import main as train_main
from rebuild_gpt.text_checkpoint import load_text_checkpoint


def args(tmp_path):
    corpus = tmp_path / "tiny.txt"
    corpus.write_text("ab" * 120, encoding="utf-8")
    return [
        "--corpus", str(corpus), "--steps", "2",
        "--block-size", "3", "--n-embd", "8", "--n-head", "2",
        "--n-layer", "1", "--batch-size", "3",
        "--eval-every", "2", "--eval-batches", "1",
        "--device", "cpu",
        "--output", str(tmp_path / "final.pt"),
        "--best-output", str(tmp_path / "best.pt"),
        "--history-file", str(tmp_path / "history.json"),
    ]


def test_bf16_cli_writes_report_and_loadable_inference_model(tmp_path):
    assert train_main(args(tmp_path) + ["--precision", "bf16"]) == 0
    report = json.loads((tmp_path / "history.json").read_text())
    assert report["requested_precision"] == "bf16"
    assert report["training_controls"]["precision"] == "bf16"
    assert [x["step"] for x in report["evaluations"]] == [0, 2]
    model, tok = load_text_checkpoint(tmp_path / "final.pt")
    assert tok.vocab_size == 2
    assert next(model.parameters()).dtype == torch.float32


def test_bf16_cli_can_resume_from_training_state(tmp_path):
    checkpoint = tmp_path / "resume.pt"
    opts = args(tmp_path) + ["--precision", "bf16"]
    assert train_main(opts + ["--training-state", str(checkpoint)]) == 0
    opts[opts.index("--steps") + 1] = "4"
    assert train_main(opts + [
        "--resume-from", str(checkpoint), "--training-state", str(checkpoint)
    ]) == 0
    report = json.loads((tmp_path / "history.json").read_text())
    assert [x["step"] for x in report["evaluations"]] == [0, 2, 4]


def test_cpu_fp16_cli_reports_clear_error(tmp_path, capsys):
    with pytest.raises(SystemExit) as error:
        train_main(args(tmp_path) + ["--precision", "fp16"])
    assert error.value.code == 2
    assert "requires CUDA" in capsys.readouterr().err
