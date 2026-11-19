"""Read-only verification of the 300-commit learning milestone evidence."""

import json
from pathlib import Path

from rebuild_gpt.resume_artifacts import sha256_file
from rebuild_gpt.review_evidence_verification import verify_review_evidence


def verify_release_evidence(corpus_path: str | Path, output_dir: str | Path) -> dict:
    """Check the entire train -> conformance -> reproduction evidence chain.

    This checks JSON and SHA-256, without executing model code or loading a
    checkpoint. It is evidence consistency, not a digital signature or an
    independent regeneration of the measurements.
    """
    folder = Path(output_dir)
    training = verify_review_evidence(corpus_path, folder)
    if training.get("status") != "verified" or training.get("conformance_checks") != 8:
        raise ValueError("Training and eight-check conformance must first verify")
    reproduction = json.loads(
        (folder / "reproduction.json").read_text(encoding="utf-8")
    )
    if (not isinstance(reproduction, dict)
            or reproduction.get("format_version") != 1
            or reproduction.get("status") != "reproduced"):
        raise ValueError("Release requires a successful reproduction report")
    source = folder / "extended_conformance.json"
    if reproduction.get("source_conformance_sha256") != sha256_file(source):
        raise ValueError("Reproduction references different conformance report bytes")
    for key in ("checkpoint_sha256", "corpus_sha256"):
        if reproduction.get(key) != training[key]:
            raise ValueError(f"Reproduction references different {key}")
    if type(reproduction.get("training_steps")) is not int or (
        reproduction["training_steps"] != training["training_steps"]
    ):
        raise ValueError("Reproduction training step does not match capstone")
    if reproduction.get("reproduced_checks") != 8:
        raise ValueError("Reproduction did not independently check all eight invariants")
    count = reproduction.get("compared_measurements")
    if type(count) is not int or count < 8:
        raise ValueError("Reproduction did not compare enough measurements")
    return {
        "status": "verified",
        "training_steps": training["training_steps"],
        "checked_invariants": 8,
        "compared_measurements": count,
        "checkpoint_sha256": training["checkpoint_sha256"],
        "corpus_sha256": training["corpus_sha256"],
        "source_conformance_sha256": reproduction["source_conformance_sha256"],
        "reproduction_report_sha256": sha256_file(folder / "reproduction.json"),
    }
