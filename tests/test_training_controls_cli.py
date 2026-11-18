import json

import pytest

from rebuild_gpt.text_checkpoint import load_text_checkpoint
from rebuild_gpt.train_cli import main


def args(tmp_path):
    corpus = tmp_path / "ab.txt"
    corpus.write_text("ab" * 100, encoding="utf-8")
    return [
        "--corpus", str(corpus), "--steps", "4",
        "--block-size", "3", "--n-embd", "8", "--n-head", "2", "--n-layer", "1",
        "--batch-size", "4", "--eval-every", "2", "--seed", "9",
        "--output", str(tmp_path / "final.pt"), "--best-output", str(tmp_path / "best.pt"),
        "--history-file", str(tmp_path / "history.json"),
        "--validation-mode", "full", "--grad-clip", "1.0",
        "--warmup-steps", "1", "--lr-decay-steps", "3", "--min-lr", "0.001",
    ]


def test_cli_full_validation_with_schedule_and_clipping(tmp_path, capsys):
    params = args(tmp_path)
    assert main(params) == 0
    capsys.readouterr()
    history = json.loads((tmp_path / "history.json").read_text(encoding="utf-8"))
    assert history["validation_mode"] == "full"
    assert history["training_controls"]["grad_clip"] == 1.0
    assert history["training_controls"]["warmup_steps"] == 1
    assert [row["step"] for row in history["evaluations"]] == [0, 2, 4]
    assert history["metrics"]["best_validation_loss"] == min(r["validation_loss"] for r in history["evaluations"])
    assert "learning_rate_last" in history["metrics"]
    model, tok = load_text_checkpoint(tmp_path / "best.pt")
    assert model.config.vocab_size == tok.vocab_size


def test_full_mode_cli_reproduces_history_exactly(tmp_path, capsys):
    params = args(tmp_path)
    assert main(params) == 0
    capsys.readouterr()
    first = (tmp_path / "history.json").read_bytes()
    assert main(params) == 0
    capsys.readouterr()
    assert first == (tmp_path / "history.json").read_bytes()


def test_cli_rejects_bad_warmup_schedule(tmp_path, capsys):
    params = args(tmp_path)
    params[params.index("--warmup-steps") + 1] = "4"
    with pytest.raises(SystemExit) as exc:
        main(params)
    assert exc.value.code == 2
    assert "decay_steps" in capsys.readouterr().err
