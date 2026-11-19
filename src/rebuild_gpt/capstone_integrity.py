"""Verify capstone training artifacts without deserializing model checkpoints."""

import json
from pathlib import Path
import re

from rebuild_gpt.resume_artifacts import verify_digest


_SHA256 = re.compile(r"[0-9a-f]{64}\Z")


def verify_capstone_artifacts(corpus_path: str | Path, output_dir: str | Path) -> dict:
    """Check exact corpus, checkpoint, and sample bytes from a trusted manifest.

    This is a read-only check; it does not call torch.load, execute model
    source, train, or regenerate text. It does not authenticate the manifest.
    """
    folder = Path(output_dir)
    manifest = json.loads((folder / "run.json").read_text(encoding="utf-8"))
    if (not isinstance(manifest, dict)
            or manifest.get("format_version") != 1
            or manifest.get("checkpoint") != "model.pt"):
        raise ValueError("Unsupported capstone manifest or checkpoint name")
    for key in ("corpus_sha256", "checkpoint_sha256"):
        digest = manifest.get(key)
        if not isinstance(digest, str) or _SHA256.fullmatch(digest) is None:
            raise ValueError(f"Malformed capstone {key}")
    settings = manifest.get("settings")
    if not isinstance(settings, dict) or not isinstance(settings.get("seed"), int):
        raise ValueError("Capstone settings must record a training seed")
    sample = manifest.get("generated_text")
    if not isinstance(sample, str):
        raise ValueError("Capstone manifest must include generated text")
    if not isinstance(manifest.get("metrics"), dict):
        raise ValueError("Capstone manifest must include training metrics")

    corpus_digest = verify_digest(
        corpus_path, manifest["corpus_sha256"], "Capstone corpus",
    )
    checkpoint_digest = verify_digest(
        folder / "model.pt", manifest["checkpoint_sha256"], "Capstone model",
    )
    sample_path = folder / "sample.txt"
    if not sample_path.is_file():
        raise FileNotFoundError(f"Capstone sample is missing: {sample_path}")
    if sample_path.read_text(encoding="utf-8") != sample:
        raise ValueError("Capstone sample differs from saved manifest")
    return {
        "status": "verified",
        "corpus_sha256": corpus_digest,
        "checkpoint_sha256": checkpoint_digest,
        "sample_bytes": sample_path.stat().st_size,
    }
