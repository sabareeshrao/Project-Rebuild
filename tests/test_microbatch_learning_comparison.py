"""Learning comparisons must control dataset, RNG, and optimizer update count."""

import json
import subprocess
import sys
from pathlib import Path

import pytest

from rebuild_gpt.microbatches import microbatch_sizes


ROOT = Path(__file__).resolve().parents[1]


def execute(tmp_path, *extra):
    corpus = tmp_path / "corpus.txt"
    corpus.write_text("ab" * 120, encoding="utf-8")
    output = tmp_path / "learning.json"
    args = [
        sys.executable, "examples/98_compare_microbatch_learning.py",
        "--corpus", str(corpus), "--steps", "2",
        "--batch-size", "3", "--block-size", "3",
        "--microbatch-sizes", "1,2,3", "--eval-every", "1",
        "--json", str(output),
        *extra,
    ]
    result = subprocess.run(
        args, cwd=ROOT, capture_output=True, text=True, timeout=90,
    )
    return result, output


def test_outcomes_comparison_records_equivalent_validation_and_weights(tmp_path):
    result, path = execute(tmp_path)
    assert result.returncode == 0, result.stderr
    data = json.loads(path.read_text(encoding="utf-8"))
    assert data["schema_version"] == 1
    assert data["settings"]["batch_size"] == 3
    assert data["settings"]["microbatch_sizes"] == [1, 2, 3]
    assert data["vocab_size"] == 2
    assert [row["microbatch_size"] for row in data["microbatch_results"]] == [1, 2, 3]
    assert [row["chunks_per_step"] for row in data["microbatch_results"]] == [
        len(microbatch_sizes(3, size)) for size in (1, 2, 3)
    ]
    expected_steps = [0, 1, 2]
    assert [entry["step"] for entry in data["baseline"]["validation_history"]] == expected_steps
    for row in data["microbatch_results"]:
        assert [entry["step"] for entry in row["validation_history"]] == expected_steps
        assert row["max_parameter_absolute_difference"] < 1e-4
        assert row["max_validation_loss_absolute_difference"] < 1e-4
        assert row["metrics"]["validation_after"] == pytest.approx(
            data["baseline"]["metrics"]["validation_after"], abs=1e-4
        )


def test_comparison_refuses_oversized_chunk_before_training(tmp_path):
    result, output = execute(tmp_path, "--microbatch-sizes", "4")
    assert result.returncode == 2
    assert "must not exceed" in result.stderr
    assert not output.exists()
