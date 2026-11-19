"""Opt-in, SHA-pinned parity report for one AdamW step.

Imports the caller's original model.py: review that file before use.
"""

from hashlib import sha256
import json
import math
from pathlib import Path
import platform
import re

import torch

from rebuild_gpt.saved_training_parity import compare_saved_capstone_adamw


def _hash_bytes(path: str | Path) -> str:
    return sha256(Path(path).read_bytes()).hexdigest()


def run_training_step_audit(
    checkpoint_path: str | Path,
    corpus_path: str | Path,
    upstream_model_path: str | Path,
    *,
    expected_upstream_sha256: str | None = None,
    output_path: str | Path | None = None,
    batch_size: int = 2,
    seed: int = 29,
    learning_rate: float = 0.001,
    weight_decay: float = 0.01,
) -> dict:
    """Write an audit only after gradients AND updated weights agree."""
    if expected_upstream_sha256 is not None:
        if not isinstance(expected_upstream_sha256, str) or re.fullmatch(
            r"[0-9a-fA-F]{64}", expected_upstream_sha256
        ) is None:
            raise ValueError("expected_upstream_sha256 must be 64 hexadecimal digits")
    source_digest = _hash_bytes(upstream_model_path)
    if (expected_upstream_sha256 is not None
            and source_digest != expected_upstream_sha256.lower()):
        raise ValueError("Trusted upstream model SHA-256 mismatch")

    checkpoint_digest = _hash_bytes(checkpoint_path)
    corpus_digest = _hash_bytes(corpus_path)
    comparison = compare_saved_capstone_adamw(
        checkpoint_path, corpus_path, upstream_model_path,
        batch_size=batch_size, seed=seed,
        learning_rate=learning_rate, weight_decay=weight_decay,
    )
    # Explicit guard: a comparison helper must not silently report NaN/Inf.
    for key in ("max_gradient_error", "max_updated_weight_error",
                "largest_local_parameter_change"):
        if not math.isfinite(comparison[key]):
            raise ValueError(f"Non-finite training parity metric: {key}")
    report = {
        "format_version": 1,
        "status": "passed",
        "comparison": comparison,
        "sha256": {
            "upstream_model_py": source_digest,
            "checkpoint": checkpoint_digest,
            "corpus": corpus_digest,
        },
        "environment": {
            "python": platform.python_version(),
            "torch": str(torch.__version__),
        },
        "scope": "one shared-weight plain PyTorch AdamW step on one held-out batch",
    }
    if output_path is not None:
        destination = Path(output_path)
        destination.parent.mkdir(parents=True, exist_ok=True)
        destination.write_text(
            json.dumps(report, sort_keys=True, indent=2) + "\n",
            encoding="utf-8",
        )
    return report
