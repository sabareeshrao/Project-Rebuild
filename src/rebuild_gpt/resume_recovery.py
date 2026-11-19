"""Choose a previously verified GPT checkpoint without opening unsafe bytes."""

from pathlib import Path

from rebuild_gpt.resume_artifacts import (
    sha256_file, stage_checkpoint_path, verify_digest,
)
from rebuild_gpt.resume_manifest import load_multistage_manifest


def select_verified_restart(
    manifest_path: str | Path,
    corpus_path: str | Path,
    *,
    before_step: int | None = None,
) -> dict:
    """Choose the newest hash-matching intermediate checkpoint.

    Never select a checkpoint at or after the desired stopping point.
    A corrupted later stage may be skipped, but a changed corpus or
    untrustworthy manifest is a hard error. This does NOT load torch.
    """
    path = Path(manifest_path)
    report = load_multistage_manifest(path)
    total = report["settings"]["total_steps"]
    if before_step is None:
        before_step = total
    if type(before_step) is not int or not 1 <= before_step <= total:
        raise ValueError("before_step must be an integer between 1 and total_steps")
    verify_digest(corpus_path, report["sha256"]["corpus"], "Corpus")

    skipped = []
    for record in reversed(report["stages"]):
        step = record["step"]
        if step >= before_step:
            continue
        checkpoint = stage_checkpoint_path(path.parent, step, total)
        if not checkpoint.is_file():
            skipped.append({"step": step, "reason": "missing"})
            continue
        if sha256_file(checkpoint) != record["sha256"]:
            skipped.append({"step": step, "reason": "sha256_mismatch"})
            continue
        return {
            "checkpoint": str(checkpoint),
            "step": step,
            "target_step": before_step,
            "sha256": record["sha256"],
            "skipped_later_stages": skipped,
        }
    raise FileNotFoundError(
        f"No hash-verified intermediate checkpoint before step {before_step}"
    )
