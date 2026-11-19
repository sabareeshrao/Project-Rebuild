"""Validate the format and claims of a completed recovery audit.

This parser performs no torch deserialization and no training. A schema
check is only the first step: the next lesson verifies the actual bytes.
"""

import json
from pathlib import Path
import re


_SHA256 = re.compile(r"[0-9a-f]{64}\Z")


def load_recovery_audit(path: str | Path) -> dict:
    """Reject inconsistent or malformed recovery audit claims."""
    audit = json.loads(Path(path).read_text(encoding="utf-8"))
    if not isinstance(audit, dict) or audit.get("format_version") != 1:
        raise ValueError("Unsupported recovery audit format")
    if audit.get("status") != "passed":
        raise ValueError("Recovery audit is not a passing result")
    for key in ("source_manifest_sha256", "recovered_checkpoint_sha256"):
        digest = audit.get(key)
        if not isinstance(digest, str) or _SHA256.fullmatch(digest) is None:
            raise ValueError(f"{key} must be a lowercase SHA-256 digest")
    recovery = audit.get("recovery")
    if not isinstance(recovery, dict) or recovery.get("status") != "identical":
        raise ValueError("Recovery is not marked identical")
    for key in ("restart_step", "target_step"):
        if type(recovery.get(key)) is not int or recovery[key] < 1:
            raise ValueError(f"{key} must be a positive integer")
    if recovery["restart_step"] >= recovery["target_step"]:
        raise ValueError("Recovery restart must precede target step")
    comparison = recovery.get("checkpoint_comparison")
    if (not isinstance(comparison, dict)
            or comparison.get("identical") is not True
            or type(comparison.get("step")) is not int
            or comparison["step"] != recovery["target_step"]):
        raise ValueError("Recovery final comparison does not match target step")
    if (not isinstance(recovery.get("recovered_checkpoint"), str)
            or not recovery["recovered_checkpoint"]):
        raise ValueError("Recovery lacks a checkpoint path")
    if not isinstance(recovery.get("skipped_later_stages"), list):
        raise ValueError("Recovery lacks skipped-stage records")
    return audit
