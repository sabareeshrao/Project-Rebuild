"""A matrix audit is only emitted after all recoveries are independently checked."""

import json
from pathlib import Path
from unittest.mock import patch

import pytest

from rebuild_gpt.recovery_matrix_audit import audit_recovery_matrix
from rebuild_gpt.recovery_matrix_cli import main


def _input(tmp_path):
    corpus = tmp_path / "corpus.txt"
    corpus.write_bytes(b"abc abc")
    source = tmp_path / "original"
    source.mkdir()
    (source / "multistage_audit.json").write_bytes(b"trusted manifest bytes")
    return corpus, source


def _run_fake_matrix(src, text, out, *, modes, damage_step):
    folder = Path(out)
    folder.mkdir()
    rows = []
    for mode in modes:
        case = folder / mode
        case.mkdir()
        (case / "recovery_audit.json").write_bytes(b"verified-" + mode.encode())
        (case / "recovered.pt").write_bytes(b"weights-" + mode.encode())
        rows.append({
            "mode": mode, "damage_step": 4, "fallback_step": 2,
            "verification": "verified", "directory": str(case),
        })
    return {"status": "passed", "trials": rows}


def test_successful_matrix_audit_identifies_each_artifact(tmp_path):
    corpus, source = _input(tmp_path)
    output = tmp_path / "matrix"
    with patch(
        "rebuild_gpt.recovery_matrix_audit.run_recovery_matrix",
        side_effect=_run_fake_matrix,
    ):
        report = audit_recovery_matrix(source, corpus, output)
    assert report["status"] == "passed"
    assert [x["mode"] for x in report["trials"]] == [
        "append", "truncate", "remove",
    ]
    assert all(len(x["recovery_audit_sha256"]) == 64 for x in report["trials"])
    assert all(len(x["recovered_checkpoint_sha256"]) == 64 for x in report["trials"])
    assert report == json.loads((output / "recovery_matrix_audit.json").read_text())
    assert not list(output.glob(".recovery_matrix.*.tmp"))


def test_failed_run_cannot_leave_passing_matrix_report(tmp_path):
    corpus, source = _input(tmp_path)
    output = tmp_path / "matrix"
    with patch(
        "rebuild_gpt.recovery_matrix_audit.run_recovery_matrix",
        side_effect=AssertionError("failed to replay truncated checkpoint"),
    ):
        with pytest.raises(AssertionError, match="failed to replay"):
            audit_recovery_matrix(source, corpus, output)
    assert not (output / "recovery_matrix_audit.json").exists()


def test_cli_forwards_modes_and_paths(capsys):
    report = {
        "status": "passed",
        "trials": [{"mode": "remove", "fallback_step": 2}],
    }
    with patch(
        "rebuild_gpt.recovery_matrix_cli.audit_recovery_matrix",
        return_value=report,
    ) as run:
        assert main([
            "--source-dir", "out/a", "--corpus", "story.txt",
            "--matrix-dir", "out/b", "--modes", "remove",
            "--damage-step", "4",
        ]) == 0
    run.assert_called_once_with(
        "out/a", "story.txt", "out/b",
        modes=("remove",), damage_step=4,
    )
    assert "passed" in capsys.readouterr().out


def test_cli_surfaces_existing_directory_failure(capsys):
    with patch(
        "rebuild_gpt.recovery_matrix_cli.audit_recovery_matrix",
        side_effect=FileExistsError("Matrix output already exists"),
    ):
        with pytest.raises(SystemExit) as error:
            main([])
    assert error.value.code == 2
    assert "already exists" in capsys.readouterr().err
