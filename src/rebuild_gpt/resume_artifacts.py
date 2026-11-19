"""Offline SHA-256 verification of a completed multi-restart GPT experiment."""

from hashlib import sha256
from pathlib import Path

from rebuild_gpt.resume_manifest import load_multistage_manifest


def sha256_file(path: str | Path) -> str:
    """Hash arbitrarily large files without reading all bytes into memory."""
    digest = sha256()
    with Path(path).open("rb") as file:
        for block in iter(lambda: file.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def stage_checkpoint_path(
    directory: str | Path, step: int, total_steps: int
) -> Path:
    """Recover filenames from the existing audit's fixed naming convention."""
    if type(step) is not int or not 0 < step <= total_steps:
        raise ValueError("step must be a positive integer within total_steps")
    name = (
        "multistage_final.pt" if step == total_steps
        else f"after_step_{step:04d}.pt"
    )
    return Path(directory) / name


def verify_digest(path: str | Path, expected: str, label: str) -> str:
    """Reject missing or modified artifacts; return the verified digest."""
    location = Path(path)
    if not location.is_file():
        raise FileNotFoundError(f"{label} is missing: {location}")
    actual = sha256_file(location)
    if actual != expected:
        raise ValueError(f"{label} SHA-256 mismatch: {location}")
    return actual


def verify_multistage_artifacts(
    manifest_path: str | Path, corpus_path: str | Path
) -> dict:
    """Check disk bytes only; this does not execute training or load torch."""
    path = Path(manifest_path)
    report = load_multistage_manifest(path)
    settings = report["settings"]
    directory = path.parent
    corpus = verify_digest(corpus_path, report["sha256"]["corpus"], "Corpus")
    reference = verify_digest(
        directory / "uninterrupted.pt",
        report["sha256"]["uninterrupted_checkpoint"],
        "Uninterrupted checkpoint",
    )
    stages = []
    for entry in report["stages"]:
        step = entry["step"]
        checkpoint = stage_checkpoint_path(
            directory, step, settings["total_steps"]
        )
        digest = verify_digest(checkpoint, entry["sha256"], f"Stage {step}")
        stages.append({
            "step": step, "path": str(checkpoint), "sha256": digest,
        })
    return {
        "status": "verified",
        "corpus_sha256": corpus,
        "uninterrupted_sha256": reference,
        "stages": stages,
    }
