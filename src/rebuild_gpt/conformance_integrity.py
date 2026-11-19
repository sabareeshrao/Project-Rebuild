"""Verify saved GPT conformance report digests without loading checkpoints."""

from pathlib import Path

from rebuild_gpt.conformance_report import load_conformance_report
from rebuild_gpt.resume_artifacts import verify_digest


def verify_conformance_integrity(
    report_path: str | Path,
    checkpoint_path: str | Path,
    corpus_path: str | Path,
) -> dict:
    """Check saved file bytes against the audit's exact SHA-256 digests.

    No model is instantiated; no .pt file is deserialized. The report
    must come from a source trusted by the operator, because its
    digests are NOT cryptographically signed.
    """
    report = load_conformance_report(report_path)
    checkpoint_hash = verify_digest(
        checkpoint_path, report["sha256"]["checkpoint"], "GPT checkpoint",
    )
    corpus_hash = verify_digest(
        corpus_path, report["sha256"]["corpus"], "Corpus",
    )
    return {
        "status": "verified",
        "check_count": len(report["checks"]),
        "checkpoint_sha256": checkpoint_hash,
        "corpus_sha256": corpus_hash,
        "checkpoint_bytes": Path(checkpoint_path).stat().st_size,
        "corpus_bytes": Path(corpus_path).stat().st_size,
    }
