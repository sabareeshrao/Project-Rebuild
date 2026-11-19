"""The CLI exposes cached sampled evaluation without changing default runs."""

import json

import pytest

from rebuild_gpt.train_cli import main as train_main


def make_corpus(tmp_path):
    path = tmp_path / "tiny.txt"
    path.write_text("abc" * 120, encoding="utf-8")
    return path


def args(tmp_path, corpus, name, *, steps="2"):
    base = tmp_path / name
    return [
        "--corpus", str(corpus), "--steps", steps,
        "--block-size", "3", "--n-embd", "8",
        "--n-head", "2", "--n-layer", "1",
        "--batch-size", "3", "--seed", "165",
        "--eval-every", "1", "--eval-batches", "2",
        "--attention-impl", "sdpa",
        "--output", str(base / "final.pt"),
        "--best-output", str(base / "best.pt"),
        "--history-file", str(base / "history.json"),
    ]


def test_default_and_cached_cli_have_identical_training_results(tmp_path):
    corpus = make_corpus(tmp_path)
    original_args = args(tmp_path, corpus, "original")
    cached_args = args(tmp_path, corpus, "cached")
    assert train_main(original_args) == 0
    assert train_main(cached_args + ["--eval-batch-backend", "cached"]) == 0
    original = json.loads(
        (tmp_path / "original" / "history.json").read_text(encoding="utf-8")
    )
    cached = json.loads(
        (tmp_path / "cached" / "history.json").read_text(encoding="utf-8")
    )
    assert original["training_controls"]["eval_batch_backend"] == "regenerate"
    assert cached["training_controls"]["eval_batch_backend"] == "cached"
    assert original["evaluations"] == cached["evaluations"]
    assert original["metrics"] == cached["metrics"]


def test_checkpoint_resume_rejects_switching_eval_backend(tmp_path):
    corpus = make_corpus(tmp_path)
    saved = tmp_path / "state.pt"
    original = args(tmp_path, corpus, "first") + [
        "--eval-batch-backend", "cached", "--training-state", str(saved),
    ]
    assert train_main(original) == 0
    mismatch = args(tmp_path, corpus, "later", steps="3") + [
        "--resume-from", str(saved),
    ]
    with pytest.raises(SystemExit) as error:
        train_main(mismatch)
    assert error.value.code == 2
    assert train_main(
        mismatch + ["--eval-batch-backend", "cached"]
    ) == 0


def test_cached_eval_rejects_full_corpus_validation(tmp_path):
    corpus = make_corpus(tmp_path)
    with pytest.raises(SystemExit) as error:
        train_main(
            args(tmp_path, corpus, "invalid") + [
                "--validation-mode", "full",
                "--eval-batch-backend", "cached",
            ]
        )
    assert error.value.code == 2
