"""Reproduce eight conformance measurements tied to a completed training run."""

from hashlib import sha256
import json
from pathlib import Path

from rebuild_gpt.conformance_reproduction import reproduce_conformance_report
from rebuild_gpt.review_evidence_verification import verify_review_evidence


def reproduce_review_evidence(corpus_path: str | Path, output_dir: str | Path) -> dict:
    """Write a new passing report only after rechecking the saved GPT.

    The original training artifacts and extended conformance report
    remain unchanged. The new report is removed before a rerun, so a
    failure cannot leave an obsolete successful result behind.
    """
    folder = Path(output_dir)
    output_path = folder / "reproduction.json"
    output_path.unlink(missing_ok=True)
    verified = verify_review_evidence(corpus_path, folder)
    if verified.get("status") != "verified" or verified.get("conformance_checks") != 8:
        raise AssertionError("Training and conformance evidence were not fully verified")
    conformance_path = folder / "extended_conformance.json"
    compared = reproduce_conformance_report(
        conformance_path, folder / "capstone" / "model.pt", corpus_path,
    )
    if compared.get("status") != "reproduced" or compared.get("reproduced_checks") != 8:
        raise AssertionError("All eight GPT measurements were not reproduced")
    if (compared.get("checkpoint_sha256") != verified["checkpoint_sha256"]
            or compared.get("corpus_sha256") != verified["corpus_sha256"]):
        raise AssertionError("Reproduction used different checkpoint or corpus bytes")
    report = {
        "format_version": 1,
        "status": "reproduced",
        "scope": "independent CPU replay of eight checks on the trained GPT",
        "source_conformance_sha256": sha256(conformance_path.read_bytes()).hexdigest(),
        "checkpoint_sha256": verified["checkpoint_sha256"],
        "corpus_sha256": verified["corpus_sha256"],
        "training_steps": verified["training_steps"],
        "reproduced_checks": compared["reproduced_checks"],
        "compared_measurements": compared["compared_measurements"],
    }
    output_path.write_text(
        json.dumps(report, sort_keys=True, indent=2) + "\n",
        encoding="utf-8",
    )
    return report
