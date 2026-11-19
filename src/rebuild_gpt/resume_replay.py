"""Resume a verified GPT training stage and compare with uninterrupted state."""

from pathlib import Path

from rebuild_gpt.resume_artifacts import verify_digest
from rebuild_gpt.resume_comparison import compare_resume_checkpoints
from rebuild_gpt.resume_manifest import load_multistage_manifest
from rebuild_gpt.resume_multistage import resume_lab_options
from rebuild_gpt.resume_recovery import select_verified_restart
from rebuild_gpt.text_training import train_char_gpt


def replay_verified_restart(
    manifest_path: str | Path,
    corpus_path: str | Path,
) -> dict:
    """Recover to the original final step using an audited intermediate point.

    The original uninterrupted checkpoint must also match its recorded
    digest, so the reference for comparison cannot have changed.
    Only load checkpoints that come from a trusted training run.
    """
    manifest = Path(manifest_path)
    report = load_multistage_manifest(manifest)
    total = report["settings"]["total_steps"]
    seed = report["settings"]["seed"]
    selected = select_verified_restart(manifest, corpus_path)
    directory = manifest.parent

    original = directory / "uninterrupted.pt"
    verify_digest(
        original, report["sha256"]["uninterrupted_checkpoint"],
        "Uninterrupted checkpoint",
    )
    recovered = directory / "recovered.pt"
    # Do not leave a result from a previous successful replay behind.
    recovered.unlink(missing_ok=True)
    history: list[dict] = []
    _, _, metrics = train_char_gpt(
        corpus_path,
        steps=total,
        history=history,
        resume_from=selected["checkpoint"],
        training_state_path=recovered,
        **resume_lab_options(total, seed),
    )
    comparison = compare_resume_checkpoints(original, recovered)
    if comparison["identical"] is not True:
        raise AssertionError("Recovered state differs from uninterrupted run")
    return {
        "status": "identical",
        "recovered_checkpoint": str(recovered),
        "restarted_from": selected["checkpoint"],
        "restart_step": selected["step"],
        "target_step": total,
        "skipped_later_stages": selected["skipped_later_stages"],
        "checkpoint_comparison": comparison,
        "validation_metrics": metrics,
        "evaluation_records": len(history),
    }
