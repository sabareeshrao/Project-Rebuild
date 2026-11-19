"""Parse a completed multistage training audit before trusting its contents.

A JSON file claiming "passed" is not evidence by itself. The following
checks establish the *shape* and internal consistency of that claim.
"""

import json
from pathlib import Path
import re

from rebuild_gpt.resume_milestones import training_milestones


_SHA256 = re.compile(r"[0-9a-f]{64}\Z")


def load_multistage_manifest(path: str | Path) -> dict:
    """Validate a trusted audit's schema, stage order, and SHA-256 digests."""
    report = json.loads(Path(path).read_text(encoding="utf-8"))
    if not isinstance(report, dict) or report.get("format_version") != 1:
        raise ValueError("Unsupported multi-stage audit format")
    if report.get("status") != "passed":
        raise ValueError("Audit does not report completed verification")
    settings = report.get("settings")
    if not isinstance(settings, dict):
        raise ValueError("Audit lacks training settings")
    total = settings.get("total_steps")
    stops = settings.get("stops")
    seed = settings.get("seed")
    if type(seed) is not int or seed < 0:
        raise ValueError("Audit seed must be a nonnegative integer")
    expected = training_milestones(total, stops)
    stages = report.get("stages")
    if not isinstance(stages, list) or len(stages) != len(expected):
        raise ValueError("Audit stages do not match interruption schedule")
    for stage, step in zip(stages, expected):
        if not isinstance(stage, dict) or type(stage.get("step")) is not int:
            raise ValueError("Audit stage is missing a valid step")
        if stage["step"] != step or not isinstance(stage.get("sha256"), str):
            raise ValueError("Audit stage step or digest disagrees with schedule")
        if _SHA256.fullmatch(stage["sha256"]) is None:
            raise ValueError("Audit stage SHA-256 must be lowercase hexadecimal")
    hashes = report.get("sha256")
    if not isinstance(hashes, dict):
        raise ValueError("Audit is missing SHA-256 identifiers")
    for name in ("corpus", "uninterrupted_checkpoint"):
        digest = hashes.get(name)
        if not isinstance(digest, str) or _SHA256.fullmatch(digest) is None:
            raise ValueError(f"Audit {name} SHA-256 is malformed")
    comparison = report.get("checkpoint_comparison")
    if not isinstance(comparison, dict) or comparison.get("identical") is not True:
        raise ValueError("Audit does not record identical final training state")
    if type(comparison.get("step")) is not int or comparison["step"] != total:
        raise ValueError("Audit final checkpoint step does not match total_steps")
    return report
