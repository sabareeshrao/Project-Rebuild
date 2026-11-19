"""Read-only triage of audited GPT training restart points."""

from pathlib import Path

from rebuild_gpt.resume_artifacts import sha256_file, stage_checkpoint_path, verify_digest
from rebuild_gpt.resume_manifest import load_multistage_manifest


def inspect_recovery_health(manifest_path: str | Path, corpus_path: str | Path) -> dict:
    """Report surviving intermediate checkpoints without loading PyTorch.

    The corpus and uninterrupted reference must pass their digest checks.
    Individual intermediate stages can be skipped if missing or corrupt.
    """
    manifest = Path(manifest_path)
    report = load_multistage_manifest(manifest)
    verify_digest(corpus_path, report["sha256"]["corpus"], "Corpus")
    verify_digest(
        manifest.parent / "uninterrupted.pt",
        report["sha256"]["uninterrupted_checkpoint"],
        "Uninterrupted checkpoint",
    )
    total = report["settings"]["total_steps"]
    stages = []
    for stage in report["stages"]:
        step = stage["step"]
        if step == total:
            continue  # The final artifact is not an intermediate restart.
        location = stage_checkpoint_path(manifest.parent, step, total)
        status = (
            "missing" if not location.is_file()
            else "verified" if sha256_file(location) == stage["sha256"]
            else "sha256_mismatch"
        )
        stages.append({"step": step, "status": status, "checkpoint": str(location)})
    usable = [stage for stage in stages if stage["status"] == "verified"]
    return {
        "status": ("unrecoverable" if not usable else
                   "ready" if len(usable) == len(stages) else "degraded"),
        "target_step": total,
        "recommended_restart_step": usable[-1]["step"] if usable else None,
        "verified_intermediates": len(usable),
        "total_intermediates": len(stages),
        "stages": stages,
    }
