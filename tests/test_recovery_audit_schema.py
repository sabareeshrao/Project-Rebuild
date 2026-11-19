"""Never take a passing recovery JSON string at face value."""

import copy
import json

import pytest

from rebuild_gpt.recovery_audit_schema import load_recovery_audit


def _audit():
    return {
        "format_version": 1, "status": "passed",
        "source_manifest_sha256": "a" * 64,
        "recovered_checkpoint_sha256": "b" * 64,
        "recovery": {
            "status": "identical",
            "restart_step": 2, "target_step": 6,
            "recovered_checkpoint": "out/recovered.pt",
            "skipped_later_stages": [],
            "checkpoint_comparison": {"identical": True, "step": 6},
        },
    }


def _read(tmp_path, payload):
    path = tmp_path / "recovery_audit.json"
    path.write_text(json.dumps(payload), encoding="utf-8")
    return load_recovery_audit(path)


def test_well_formed_recovery_audit_is_accepted(tmp_path):
    original = _audit()
    assert _read(tmp_path, original) == original


@pytest.mark.parametrize("change", [
    lambda a: a.update(status="failed"),
    lambda a: a.update(format_version=17),
    lambda a: a.update(source_manifest_sha256="not-a-hash"),
    lambda a: a.update(recovered_checkpoint_sha256="A" * 64),
    lambda a: a["recovery"].update(status="failed"),
    lambda a: a["recovery"].update(restart_step=6),
    lambda a: a["recovery"].update(target_step=True),
    lambda a: a["recovery"].update(recovered_checkpoint=""),
    lambda a: a["recovery"].update(skipped_later_stages=None),
    lambda a: a["recovery"]["checkpoint_comparison"].update(identical=False),
    lambda a: a["recovery"]["checkpoint_comparison"].update(step=5),
])
def test_forged_or_inconsistent_claims_are_rejected(tmp_path, change):
    candidate = copy.deepcopy(_audit())
    change(candidate)
    with pytest.raises(ValueError):
        _read(tmp_path, candidate)
