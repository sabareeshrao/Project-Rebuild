"""Several seeds must pass independently and never inherit stale success."""

import json
from unittest.mock import patch

import pytest

from rebuild_gpt.conformance_seed_sweep import run_conformance_seed_sweep


def _fake(checkpoint, corpus, *, seed, batch_size):
    return {
        "status": "passed",
        "model": {"n_layer": 1, "vocab_size": 5, "attention_impl": "fused"},
        "sha256": {"checkpoint": "a" * 64, "corpus": "b" * 64},
        "checks": {f"check{i}": {"seed": seed} for i in range(8)},
    }


def test_distinct_seed_runs_use_same_checkpoint(tmp_path):
    dest = tmp_path / "sweep.json"
    with patch(
        "rebuild_gpt.conformance_seed_sweep.run_extended_conformance",
        side_effect=_fake,
    ) as run:
        result = run_conformance_seed_sweep(
            "trained.pt", "story.txt", seeds=(11, 29, 47),
            batch_size=3, output_path=dest,
        )
    assert run.call_count == 3
    assert result["status"] == "passed"
    assert [row["seed"] for row in result["rounds"]] == [11, 29, 47]
    assert result["rounds"][0]["checks"] != result["rounds"][1]["checks"]
    assert json.loads(dest.read_text(encoding="utf-8")) == result


@pytest.mark.parametrize("seeds", [
    (), (11,), (3, 3), (-1, 7), (True, 3), (1.5, 3), "1,2",
])
def test_invalid_seed_schedule_fails_before_running(seeds):
    with pytest.raises(ValueError, match="distinct"):
        run_conformance_seed_sweep("missing.pt", "missing.txt", seeds=seeds)


def test_one_failed_seed_prevents_passing_report(tmp_path):
    dest = tmp_path / "sweep.json"
    dest.write_text('{"status": "passed"}', encoding="utf-8")

    def fail_on_second(*args, **kwargs):
        if kwargs["seed"] == 29:
            raise AssertionError("causal masking failed")
        return _fake(*args, **kwargs)

    with patch(
        "rebuild_gpt.conformance_seed_sweep.run_extended_conformance",
        side_effect=fail_on_second,
    ):
        with pytest.raises(AssertionError, match="causal masking"):
            run_conformance_seed_sweep(
                "trained.pt", "story.txt", seeds=(11, 29), output_path=dest,
            )
    assert not dest.exists()


def test_drifted_checkpoint_identity_is_not_accepted():
    def drift(*args, **kwargs):
        report = _fake(*args, **kwargs)
        if kwargs["seed"] == 29:
            report["sha256"]["checkpoint"] = "c" * 64
        return report

    with patch(
        "rebuild_gpt.conformance_seed_sweep.run_extended_conformance",
        side_effect=drift,
    ):
        with pytest.raises(AssertionError, match="changed"):
            run_conformance_seed_sweep("trained.pt", "story.txt")
