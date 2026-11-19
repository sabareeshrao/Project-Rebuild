"""The CLI reports the bounded cache and preserves resume policy."""

import json

import pytest

from rebuild_gpt.train_cli import main as train_main


def make_corpus(tmp_path):
    path = tmp_path / "text.txt"
    path.write_text("abc" * 120, encoding="utf-8")
    return path


def arguments(tmp_path, corpus, folder, *, steps="1"):
    root = tmp_path / folder
    return [
        "--corpus", str(corpus), "--steps", steps,
        "--block-size", "3", "--n-embd", "8", "--n-head", "2",
        "--n-layer", "1", "--batch-size", "3",
        "--eval-every", "1", "--eval-batches", "2",
        "--attention-impl", "sdpa", "--seed", "170",
        "--eval-batch-backend", "cached",
        "--output", str(root / "model.pt"),
        "--best-output", str(root / "best.pt"),
        "--history-file", str(root / "history.json"),
    ]


def test_cli_budget_reports_logical_bytes_and_preserves_scores(tmp_path):
    corpus = make_corpus(tmp_path)
    base = arguments(tmp_path, corpus, "without_limit")
    capped = arguments(tmp_path, corpus, "with_limit")
    bytes_needed = 2 * 2 * 3 * 3 * 2 * 8
    cap_mib = str(bytes_needed / (1024 * 1024))
    assert train_main(base) == 0
    assert train_main(capped + ["--eval-cache-max-mib", cap_mib]) == 0
    original = json.loads(
        (tmp_path / "without_limit/history.json").read_text(encoding="utf-8")
    )
    bounded = json.loads(
        (tmp_path / "with_limit/history.json").read_text(encoding="utf-8")
    )
    assert bounded["training_controls"]["eval_cache_estimated_logical_bytes"] == bytes_needed
    assert bounded["training_controls"]["eval_cache_max_mib"] == float(cap_mib)
    assert original["evaluations"] == bounded["evaluations"]
    assert original["metrics"] == bounded["metrics"]


def test_cli_rejects_budget_below_estimate(tmp_path):
    text = make_corpus(tmp_path)
    args = arguments(tmp_path, text, "too_small") + [
        "--eval-cache-max-mib", "0.00001",
    ]
    with pytest.raises(SystemExit) as error:
        train_main(args)
    assert error.value.code == 2
    assert not (tmp_path / "too_small/history.json").exists()


def test_cli_rejects_cache_limit_with_regenerate_mode(tmp_path):
    text = make_corpus(tmp_path)
    args = arguments(tmp_path, text, "invalid")
    index = args.index("--eval-batch-backend")
    args[index + 1] = "regenerate"
    with pytest.raises(SystemExit) as error:
        train_main(args + ["--eval-cache-max-mib", "1"])
    assert error.value.code == 2


def test_cli_resume_rejects_changed_budget(tmp_path):
    text = make_corpus(tmp_path)
    checkpoint = tmp_path / "state.pt"
    initial = arguments(tmp_path, text, "initial", steps="1") + [
        "--eval-cache-max-mib", "0.001", "--training-state", str(checkpoint),
    ]
    assert train_main(initial) == 0
    mismatch = arguments(tmp_path, text, "resumed", steps="2") + [
        "--resume-from", str(checkpoint),
    ]
    with pytest.raises(SystemExit) as error:
        train_main(mismatch)
    assert error.value.code == 2
    assert train_main(
        mismatch + ["--eval-cache-max-mib", "0.001"]
    ) == 0
