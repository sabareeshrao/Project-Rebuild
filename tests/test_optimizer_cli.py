import json

import pytest
import torch

from rebuild_gpt.text_checkpoint import load_text_checkpoint
from rebuild_gpt.text_cli import main as generate_main
from rebuild_gpt.train_cli import main as train_main


def args(tmp_path):
    path = tmp_path / "ab.txt"
    path.write_text("ab" * 130, encoding="utf-8")
    return [
        "--corpus", str(path), "--steps", "5", "--block-size", "4",
        "--batch-size", "3", "--n-embd", "8", "--n-head", "2", "--n-layer", "1",
        "--eval-every", "2", "--eval-batches", "2", "--seed", "8",
        "--grouped-adamw", "--weight-decay", "0.2", "--adam-beta1", "0.8",
        "--adam-beta2", "0.9", "--gradient-accumulation-steps", "3",
        "--grad-clip", "0.5", "--warmup-steps", "1",
        "--lr-decay-steps", "4", "--min-lr", "0.001",
        "--output", str(tmp_path / "final.pt"),
        "--best-output", str(tmp_path / "best.pt"),
        "--history-file", str(tmp_path / "history.json"),
        "--restore-best",
    ]


def test_cli_writes_optimizer_metadata_and_loadable_best_model(tmp_path, capsys):
    assert train_main(args(tmp_path)) == 0
    capsys.readouterr()
    report = json.loads((tmp_path / "history.json").read_text())
    controls = report["training_controls"]
    assert controls["gradient_accumulation_steps"] == 3
    assert controls["grouped_adamw"] is True
    assert controls["weight_decay"] == 0.2
    assert controls["adam_betas"] == [0.8, 0.9]
    assert report["metrics"]["optimizer_group_count"] == 2
    assert report["metrics"]["effective_tokens_per_step"] == 36
    final, tok = load_text_checkpoint(tmp_path / "final.pt")
    best, _ = load_text_checkpoint(tmp_path / "best.pt")
    assert all(torch.equal(a,b) for a,b in zip(final.parameters(), best.parameters()))
    assert generate_main(["--checkpoint", str(tmp_path / "best.pt"), "--prompt", "a",
                          "--new-tokens", "3", "--seed", "11"]) == 0
    assert capsys.readouterr().out.startswith("a")


def test_optimizer_cli_is_reproducible(tmp_path, capsys):
    params = args(tmp_path)
    train_main(params)
    capsys.readouterr()
    first = (tmp_path / "history.json").read_bytes()
    train_main(params)
    capsys.readouterr()
    assert (tmp_path / "history.json").read_bytes() == first


@pytest.mark.parametrize("switch,value,match", [
    ("--gradient-accumulation-steps", "0", "gradient_accumulation_steps"),
    ("--weight-decay", "-1", "weight_decay"),
    ("--adam-beta2", "1.0", "betas"),
])
def test_cli_rejects_invalid_optimizer_options(tmp_path, capsys, switch, value, match):
    options = args(tmp_path)
    options[options.index(switch) + 1] = value
    with pytest.raises(SystemExit) as exc:
        train_main(options)
    assert exc.value.code == 2
    assert match in capsys.readouterr().err
