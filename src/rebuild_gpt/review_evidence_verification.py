"""Connect capstone training evidence to the exact saved conformance run."""

from pathlib import Path
import json
import math

from rebuild_gpt.capstone_integrity import verify_capstone_artifacts
from rebuild_gpt.conformance_integrity import verify_conformance_integrity
from rebuild_gpt.conformance_report import load_conformance_report


def verify_review_evidence(corpus_path: str | Path, output_dir: str | Path) -> dict:
    """Verify both manifests refer to one model and the same corpus.

    Performs read-only JSON and SHA-256 checks, with no torch.load and no
    training. A trusted manifest is required; SHA-256 alone is not a signature.
    """
    folder = Path(output_dir)
    capstone_dir = folder / "capstone"
    capstone = json.loads(
        (capstone_dir / "run.json").read_text(encoding="utf-8")
    )
    training = verify_capstone_artifacts(corpus_path, capstone_dir)
    conformance = load_conformance_report(folder / "extended_conformance.json")
    checked = verify_conformance_integrity(
        folder / "extended_conformance.json",
        capstone_dir / "model.pt", corpus_path,
    )
    if (training["checkpoint_sha256"] != checked["checkpoint_sha256"]
            or training["corpus_sha256"] != checked["corpus_sha256"]):
        raise AssertionError("Training and conformance used different model/corpus bytes")
    spec = capstone.get("settings")
    model = conformance["model"]
    if not isinstance(spec, dict):
        raise ValueError("Capstone settings are missing")
    for key in ("block_size", "n_layer"):
        if type(spec.get(key)) is not int or spec[key] != model[key]:
            raise ValueError(f"Conformance model differs from training {key}")
    metrics = capstone.get("metrics")
    if not isinstance(metrics, dict):
        raise ValueError("Training metrics are missing")
    for key in ("validation_before", "validation_after", "train_loss_last"):
        value = metrics.get(key)
        if type(value) not in (float, int) or not math.isfinite(value):
            raise ValueError(f"Invalid capstone metric: {key}")
    return {
        "status": "verified",
        "checkpoint_sha256": checked["checkpoint_sha256"],
        "corpus_sha256": checked["corpus_sha256"],
        "conformance_checks": checked["check_count"],
        "training_steps": spec["steps"],
        "training_validation_before": metrics["validation_before"],
        "training_validation_after": metrics["validation_after"],
    }
