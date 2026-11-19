"""CLI preserves index-only evaluation choices and conservative byte reports."""

import json

import pytest

from rebuild_gpt.train_cli import main as train_main


def corpus(tmp_path):
    path = tmp_path / "corpus.txt"
    path.write_text("abc" * 120, encoding="utf-8")
    return path


def args(tmp_path, corpus_file, name):
    root = tmp_path / name
    return [
        "--corpus", str(corpus_file), "--steps", "2",
        "--block-size", "3", "--batch-size", "3",
        "--n-embd", "8", "--n-head", "2", "--n-layer", "1",
        "--seed", "177", "--eval-every", "1", "--eval-batches", "2",
        "--attention-impl", "sdpa",
        "--output", str(root / "final.pt"),
        "--best-output", str(root / "best.pt"),
        "--history-file", str(root / "history.json"),
    ]


def test_cli_index_backend_matches_default_training(tmp_path):
    path = corpus(tmp_path)
    baseline = args(tmp_path, path, "baseline")
    indexed = args(tmp_path, path, "indexed")
    assert train_main(baseline) == 0
    # 2 splits x 2 eval batches x 3 starts x 8 bytes = 96 bytes.
    assert train_main(indexed + [
        "--eval-batch-backend", "indices",
        "--eval-cache-max-mib", str(96 / 1048576),
    ]) == 0
    a = json.loads((tmp_path / "baseline/history.json").read_text())
    b = json.loads((tmp_path / "indexed/history.json").read_text())
    assert a["evaluations"] == b["evaluations"]
    assert a["metrics"] == b["metrics"]
    assert a["training_controls"]["eval_cache_estimated_logical_bytes"] is None
    assert b["training_controls"]["eval_cache_estimated_logical_bytes"] == 96
    assert b["training_controls"]["eval_batch_backend"] == "indices"


def test_cli_rejects_index_plan_under_tight_budget(tmp_path):
    path = corpus(tmp_path)
    with pytest.raises(SystemExit) as error:
        train_main(args(tmp_path, path, "small") + [
            "--eval-batch-backend", "indices", "--eval-cache-max-mib", "0.00001",
        ])
    assert error.value.code == 2


def test_indices_disallowed_with_full_corpus_evaluation(tmp_path):
    path = corpus(tmp_path)
    with pytest.raises(SystemExit) as error:
        train_main(args(tmp_path, path, "full") + [
            "--eval-batch-backend", "indices", "--validation-mode", "full",
        ])
    assert error.value.code == 2
