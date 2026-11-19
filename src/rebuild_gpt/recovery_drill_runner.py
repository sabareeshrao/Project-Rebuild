"""Exercise GPT checkpoint failure and recovery end-to-end in a copy."""

from pathlib import Path

from rebuild_gpt.recovery_audit_verification import verify_recovery_audit
from rebuild_gpt.recovery_drill import prepare_recovery_drill
from rebuild_gpt.resume_manifest import load_multistage_manifest
from rebuild_gpt.resume_recovery_cli import main as recovery_cli


def run_recovery_drill(
    source_dir: str | Path,
    corpus_path: str | Path,
    drill_dir: str | Path,
    *,
    damage_step: int,
    damage_mode: str = "append",
) -> dict:
    """Recover from one damaged latest-intermediate checkpoint in a new copy.

    Requires at least two interruption points, so an earlier intact
    checkpoint is guaranteed to exist before we inject the failure.
    """
    manifest = load_multistage_manifest(
        Path(source_dir) / "multistage_audit.json"
    )
    stops = manifest["settings"]["stops"]
    if len(stops) < 2 or type(damage_step) is not int or damage_step != stops[-1]:
        raise ValueError(
            "damage_step must be the latest intermediate stop, with an earlier fallback"
        )
    copied = prepare_recovery_drill(
        source_dir, corpus_path, drill_dir,
        damage_step=damage_step, damage_mode=damage_mode,
    )
    # This command writes recovery_audit.json only after full state parity.
    success = recovery_cli([
        "--mode", "recover", "--run-dir", str(drill_dir),
        "--corpus", str(corpus_path),
    ])
    if success != 0:
        raise AssertionError("Recovery command did not report success")
    independently_verified = verify_recovery_audit(
        Path(drill_dir) / "recovery_audit.json", corpus_path,
    )
    if independently_verified["restart_step"] >= damage_step:
        raise AssertionError("Drill did not fall back before damaged step")
    return {
        "status": "passed",
        "damage_step": damage_step,
        "damage_mode": copied["damage_mode"],
        "fallback_step": independently_verified["restart_step"],
        "original_directory": copied["original_directory"],
        "drill_directory": copied["drill_directory"],
        "independent_check": independently_verified,
    }
