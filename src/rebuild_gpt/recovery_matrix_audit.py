"""Produce a SHA-256-identified audit of independent recovery fault trials."""

from hashlib import sha256
import json
import os
from pathlib import Path
import platform
from tempfile import NamedTemporaryFile

import torch

from rebuild_gpt.recovery_matrix import FAULT_MODES, run_recovery_matrix
from rebuild_gpt.resume_artifacts import sha256_file


def audit_recovery_matrix(
    source_dir: str | Path,
    corpus_path: str | Path,
    output_dir: str | Path,
    *,
    modes: tuple[str, ...] = FAULT_MODES,
    damage_step: int | None = None,
) -> dict:
    """Save a successful summary only after EVERY isolated replay passes.

    The output directory must be new. A failed trial may leave its
    disposable debugging artifacts, but never a 'passed' matrix report.
    """
    result = run_recovery_matrix(
        source_dir, corpus_path, output_dir,
        modes=modes, damage_step=damage_step,
    )
    if result.get("status") != "passed" or len(result.get("trials", [])) != len(modes):
        raise AssertionError("Not all recovery matrix cases passed")
    output = Path(output_dir)
    trials = []
    for item in result["trials"]:
        if item.get("verification") != "verified" or item.get("mode") not in modes:
            raise AssertionError("A recovery matrix case lacks independent verification")
        location = Path(item["directory"])
        trials.append({
            "mode": item["mode"],
            "damage_step": item["damage_step"],
            "fallback_step": item["fallback_step"],
            "recovery_audit_sha256": sha256_file(location / "recovery_audit.json"),
            "recovered_checkpoint_sha256": sha256_file(location / "recovered.pt"),
        })
    if {t["mode"] for t in trials} != set(modes):
        raise AssertionError("Recovery matrix modes differ from requested modes")
    report = {
        "format_version": 1,
        "status": "passed",
        "scope": "independent CPU fp32 checkpoint recovery fault experiments",
        "source_manifest_sha256": sha256_file(
            Path(source_dir) / "multistage_audit.json"
        ),
        "corpus_sha256": sha256_file(corpus_path),
        "trials": trials,
        "environment": {
            "python": platform.python_version(),
            "torch": str(torch.__version__),
        },
    }
    destination = output / "recovery_matrix_audit.json"
    with NamedTemporaryFile(
        mode="w", encoding="utf-8", dir=output,
        prefix=".recovery_matrix.", suffix=".tmp", delete=False,
    ) as temp:
        temporary = Path(temp.name)
        try:
            json.dump(report, temp, indent=2, sort_keys=True)
            temp.write("\n")
        except BaseException:
            temporary.unlink(missing_ok=True)
            raise
    try:
        os.replace(temporary, destination)
    finally:
        temporary.unlink(missing_ok=True)
    return report
