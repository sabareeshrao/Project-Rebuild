"""Schema validation rejects forged, incomplete or reordered audit claims."""

import json

import pytest

from rebuild_gpt.resume_manifest import load_multistage_manifest


def _audit():
    return {
        "format_version": 1, "status": "passed",
        "settings": {"total_steps": 6, "stops": [2, 4], "seed": 13},
        "sha256": {
            "corpus": "a" * 64,
            "uninterrupted_checkpoint": "b" * 64,
        },
        "stages": [
            {"step": step, "sha256": "c" * 64} for step in (2, 4, 6)
        ],
        "checkpoint_comparison": {"step": 6, "identical": True},
    }


def _load(tmp_path, data):
    path = tmp_path / "audit.json"
    path.write_text(json.dumps(data), encoding="utf-8")
    return load_multistage_manifest(path)


def test_valid_audit_keeps_all_stage_metadata(tmp_path):
    expected = _audit()
    assert _load(tmp_path, expected) == expected


@pytest.mark.parametrize("change", [
    lambda a: a.update(status="failed"),
    lambda a: a["settings"].update(stops=[4, 2]),
    lambda a: a["settings"].update(total_steps=4),
    lambda a: a["settings"].update(seed=True),
    lambda a: a["stages"].pop(),
    lambda a: a["stages"][0].update(step=3),
    lambda a: a["stages"][1].update(sha256="not-a-digest"),
    lambda a: a["sha256"].update(corpus="A" * 64),
    lambda a: a["checkpoint_comparison"].update(identical=False),
])
def test_malformed_audit_is_rejected(tmp_path, change):
    report = _audit()
    change(report)
    with pytest.raises(ValueError):
        _load(tmp_path, report)


def test_invalid_json_does_not_look_like_success(tmp_path):
    path = tmp_path / "audit.json"
    path.write_text("{broken", encoding="utf-8")
    with pytest.raises(ValueError):
        load_multistage_manifest(path)
