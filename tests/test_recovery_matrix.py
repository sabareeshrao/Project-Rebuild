"""Matrix orchestration must isolate cases and require independent success."""

from hashlib import sha256
import json
from pathlib import Path
from unittest.mock import patch

import pytest

from rebuild_gpt.recovery_matrix import run_recovery_matrix


def _source(tmp_path):
    folder = tmp_path / "source"
    folder.mkdir()
    corpus = tmp_path / "corpus.txt"
    corpus.write_bytes(b"abba" * 40)
    ref = folder / "uninterrupted.pt"
    ref.write_bytes(b"trusted-reference")
    stages = []
    for step in (2, 4, 6):
        name = "multistage_final.pt" if step == 6 else f"after_step_{step:04d}.pt"
        file = folder / name
        file.write_bytes(str(step).encode())
        stages.append({"step": step, "sha256": sha256(file.read_bytes()).hexdigest()})
    manifest = {
        "format_version": 1, "status": "passed",
        "settings": {"total_steps": 6, "stops": [2, 4], "seed": 5},
        "sha256": {
            "corpus": sha256(corpus.read_bytes()).hexdigest(),
            "uninterrupted_checkpoint": sha256(ref.read_bytes()).hexdigest(),
        },
        "stages": stages,
        "checkpoint_comparison": {"step": 6, "identical": True},
    }
    (folder / "multistage_audit.json").write_text(json.dumps(manifest), encoding="utf-8")
    return folder, corpus


def test_three_modes_get_independent_directories(tmp_path):
    source, corpus = _source(tmp_path)
    output = tmp_path / "matrix"

    def fake_trial(src, text, directory, *, damage_step, damage_mode):
        assert Path(src) == source and Path(text) == corpus
        assert damage_step == 4
        return {
            "status": "passed", "fallback_step": 2,
            "independent_check": {"status": "verified"},
        }

    with patch(
        "rebuild_gpt.recovery_matrix.run_recovery_drill",
        side_effect=fake_trial,
    ) as run:
        report = run_recovery_matrix(source, corpus, output)
    assert report["status"] == "passed"
    assert run.call_count == 3
    assert [row["mode"] for row in report["trials"]] == [
        "append", "truncate", "remove",
    ]
    assert [row["directory"] for row in report["trials"]] == [
        str(output / mode) for mode in ("append", "truncate", "remove")
    ]


def test_failure_in_one_trial_cannot_return_success(tmp_path):
    source, corpus = _source(tmp_path)
    result = {"status": "passed", "fallback_step": 2,
              "independent_check": {"status": "verified"}}
    with patch(
        "rebuild_gpt.recovery_matrix.run_recovery_drill",
        side_effect=[result, {**result, "status": "failed"}],
    ):
        with pytest.raises(AssertionError, match="not independently verified"):
            run_recovery_matrix(
                source, corpus, tmp_path / "matrix", modes=("append", "remove"),
            )


@pytest.mark.parametrize("modes", [
    (), ("append", "append"), ("unknown",), "append",
])
def test_invalid_modes_do_not_create_directories(tmp_path, modes):
    source, corpus = _source(tmp_path)
    dest = tmp_path / "matrix"
    with pytest.raises(ValueError, match="modes"):
        run_recovery_matrix(source, corpus, dest, modes=modes)
    assert not dest.exists()


def test_existing_output_is_never_overwritten(tmp_path):
    source, corpus = _source(tmp_path)
    dest = tmp_path / "matrix"
    dest.mkdir()
    sentinel = dest / "keep.txt"
    sentinel.write_text("important")
    with pytest.raises(FileExistsError):
        run_recovery_matrix(source, corpus, dest)
    assert sentinel.read_text() == "important"
