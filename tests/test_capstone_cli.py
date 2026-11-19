"""Exercise the complete public CPU training-to-verification workflow."""

import json

from rebuild_gpt.capstone_cli import main


def test_capstone_cli_trains_and_verifies_without_retraining(tmp_path, capsys):
    corpus = tmp_path / "story.txt"
    corpus.write_text("ab" * 70, encoding="utf-8")
    destination = tmp_path / "capstone"
    arguments = [
        "--corpus", str(corpus), "--output-dir", str(destination),
        "--steps", "2", "--seed", "3", "--prompt", "a",
        "--new-tokens", "3",
    ]
    assert main(arguments) == 0
    first = capsys.readouterr().out
    assert "Trained, saved, and verified" in first

    report = json.loads((destination / "run.json").read_text(encoding="utf-8"))
    assert report["settings"]["steps"] == 2
    assert report["settings"]["seed"] == 3
    assert len(report["generated_text"]) == 4

    assert main([
        "--verify-only", "--corpus", str(corpus),
        "--output-dir", str(destination),
    ]) == 0
    second = capsys.readouterr().out
    assert "Verified saved checkpoint" in second
    assert "model.pt" in (destination / "run.json").read_text(encoding="utf-8")
