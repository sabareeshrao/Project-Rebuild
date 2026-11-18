import json

import pytest
import torch

from rebuild_gpt.text_checkpoint import load_text_checkpoint
from rebuild_gpt.text_cli import main as generate_main
from rebuild_gpt.train_cli import main as train_main


def options(tmp_path):
    corpus = tmp_path / "corpus.txt"
    corpus.write_text("ab" * 100, encoding="utf-8")
    return [
        "--corpus", str(corpus), "--steps", "5",
        "--block-size", "4", "--n-embd", "8", "--n-head", "2",
        "--n-layer", "1", "--batch-size", "4",
        "--eval-every", "2", "--eval-batches", "2",
        "--seed", "7",
        "--output", str(tmp_path / "final.pt"),
        "--best-output", str(tmp_path / "best.pt"),
        "--history-file", str(tmp_path / "history.json"),
    ]


def test_cli_saves_final_best_and_json_history(tmp_path, capsys):
    args = options(tmp_path)
    assert train_main(args) == 0
    assert "best validation" in capsys.readouterr().out
    report = json.loads((tmp_path / "history.json").read_text(encoding="utf-8"))
    assert report["seed"] == 7
    assert [entry["step"] for entry in report["evaluations"]] == [0, 2, 4, 5]
    assert report["metrics"]["best_validation_loss"] == min(
        entry["validation_loss"] for entry in report["evaluations"]
    )
    final, tok = load_text_checkpoint(tmp_path / "final.pt")
    best, best_tok = load_text_checkpoint(tmp_path / "best.pt")
    assert tok.stoi == best_tok.stoi
    assert final.config == best.config
    assert generate_main([
        "--checkpoint", str(tmp_path / "best.pt"), "--prompt", "a",
        "--new-tokens", "4", "--top-k", "1",
    ]) == 0
    assert capsys.readouterr().out.strip().startswith("a")


def test_restore_best_flag_makes_final_checkpoint_equal_best(tmp_path, capsys):
    assert train_main(options(tmp_path) + ["--restore-best"]) == 0
    capsys.readouterr()
    final, _ = load_text_checkpoint(tmp_path / "final.pt")
    best, _ = load_text_checkpoint(tmp_path / "best.pt")
    assert all(torch.equal(a, b) for a, b in zip(final.parameters(), best.parameters()))


def test_cli_history_is_reproducible_across_runs(tmp_path, capsys):
    args = options(tmp_path)
    train_main(args)
    capsys.readouterr()
    first = (tmp_path / "history.json").read_text(encoding="utf-8")
    train_main(args)
    capsys.readouterr()
    assert first == (tmp_path / "history.json").read_text(encoding="utf-8")


def test_cli_rejects_same_output_and_best_paths(tmp_path, capsys):
    args = options(tmp_path)
    index = args.index("--best-output") + 1
    args[index] = str(tmp_path / "final.pt")
    with pytest.raises(SystemExit) as exc:
        train_main(args)
    assert exc.value.code == 2
    assert "distinct" in capsys.readouterr().err


def test_cli_reports_invalid_training_options(tmp_path, capsys):
    args = options(tmp_path)
    index = args.index("--eval-every") + 1
    args[index] = "0"
    with pytest.raises(SystemExit) as exc:
        train_main(args)
    assert exc.value.code == 2
    assert "eval_interval" in capsys.readouterr().err
