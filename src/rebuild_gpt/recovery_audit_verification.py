"""Independently verify recovered GPT training state without re-training."""

from pathlib import Path

from rebuild_gpt.recovery_audit_schema import load_recovery_audit
from rebuild_gpt.resume_artifacts import (
    stage_checkpoint_path, verify_digest,
)
from rebuild_gpt.resume_comparison import compare_resume_checkpoints
from rebuild_gpt.resume_manifest import load_multistage_manifest


def verify_recovery_audit(
    recovery_audit_path: str | Path, corpus_path: str | Path,
) -> dict:
    """Check saved digests and compare exact model, AdamW and RNG contents.

    Only use trusted audit files and trusted PyTorch checkpoint files.
    This runs no optimizer steps, but does deserialize checkpoint tensors.
    """
    path = Path(recovery_audit_path)
    audit = load_recovery_audit(path)
    recovery = audit["recovery"]
    directory = path.parent
    source_manifest = directory / "multistage_audit.json"
    verify_digest(source_manifest, audit["source_manifest_sha256"], "Source manifest")
    source = load_multistage_manifest(source_manifest)
    final_step = source["settings"]["total_steps"]
    restart_step = recovery["restart_step"]
    if recovery["target_step"] != final_step:
        raise ValueError("Recovery target differs from the original run")
    if restart_step not in source["settings"]["stops"]:
        raise ValueError("Recovery step was not a recorded interruption")
    verify_digest(corpus_path, source["sha256"]["corpus"], "Corpus")
    interrupted = stage_checkpoint_path(directory, restart_step, final_step)
    if Path(recovery.get("restarted_from", "")).resolve() != interrupted.resolve():
        raise ValueError("Recovery source checkpoint path does not match stage")
    expected_stage = next(
        entry for entry in source["stages"] if entry["step"] == restart_step
    )
    verify_digest(interrupted, expected_stage["sha256"], "Restart checkpoint")
    recovered = directory / "recovered.pt"
    if Path(recovery["recovered_checkpoint"]).resolve() != recovered.resolve():
        raise ValueError("Recovery output path is not the expected checkpoint")
    verify_digest(recovered, audit["recovered_checkpoint_sha256"], "Recovered checkpoint")
    reference = directory / "uninterrupted.pt"
    verify_digest(
        reference, source["sha256"]["uninterrupted_checkpoint"],
        "Uninterrupted checkpoint",
    )
    result = compare_resume_checkpoints(reference, recovered)
    if result != recovery["checkpoint_comparison"]:
        raise AssertionError("Current training-state comparison differs from saved audit")
    if result["identical"] is not True or result["step"] != final_step:
        raise AssertionError("Recovered training state is not identical")
    return {
        "status": "verified",
        "restart_step": restart_step,
        "target_step": final_step,
        "checkpoint_comparison": result,
    }
