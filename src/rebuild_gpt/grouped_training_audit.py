"""SHA-256-pinned multi-step grouped AdamW comparison for a saved GPT."""

from hashlib import sha256
import json
from pathlib import Path
import platform
import re

import torch

from rebuild_gpt.capstone_parity import prepare_validation_parity_batch
from rebuild_gpt.grouped_step_parity import compare_grouped_adamw_trajectory
from rebuild_gpt.text_checkpoint import load_text_checkpoint
from rebuild_gpt.upstream_parity import load_upstream_model


def sha256_file(path: str | Path) -> str:
    """Hash exact on-disk bytes, not the filename or modification time."""
    return sha256(Path(path).read_bytes()).hexdigest()


def audit_grouped_training_trajectory(
    checkpoint_path: str | Path,
    corpus_path: str | Path,
    upstream_model_path: str | Path,
    *,
    upstream_sha256: str,
    output_path: str | Path | None = None,
    steps: int = 3,
    seed: int = 29,
    batch_size: int = 2,
    learning_rate: float = 0.001,
    weight_decay: float = 0.1,
) -> dict:
    """Never import original Python unless the exact supplied digest matches.

    Returns a passed report ONLY after the matched-weight multi-step
    gradient and parameter comparisons have succeeded.
    """
    if not isinstance(upstream_sha256, str) or re.fullmatch(
        r"[0-9a-fA-F]{64}", upstream_sha256,
    ) is None:
        raise ValueError("upstream_sha256 must be 64 hexadecimal digits")
    if sha256_file(upstream_model_path) != upstream_sha256.lower():
        raise ValueError("Trusted upstream source SHA-256 mismatch")
    checkpoint_hash = sha256_file(checkpoint_path)
    corpus_hash = sha256_file(corpus_path)

    ours, tokenizer = load_text_checkpoint(checkpoint_path)
    inputs, targets = prepare_validation_parity_batch(
        corpus_path, tokenizer, ours.config, batch_size=batch_size, seed=seed,
    )
    upstream = load_upstream_model(upstream_model_path, ours.config)
    comparison = compare_grouped_adamw_trajectory(
        ours, upstream, inputs, targets, steps=steps,
        learning_rate=learning_rate, weight_decay=weight_decay,
    )
    report = {
        "format_version": 1,
        "status": "passed",
        "scope": "matching grouped-AdamW trajectory on a repeated held-out token batch",
        "sha256": {
            "upstream_model_py": upstream_sha256.lower(),
            "checkpoint": checkpoint_hash,
            "corpus": corpus_hash,
        },
        "environment": {
            "python": platform.python_version(),
            "torch": str(torch.__version__),
        },
        "settings": {
            "steps": steps, "seed": seed, "batch_size": batch_size,
            "learning_rate": learning_rate, "weight_decay": weight_decay,
        },
        "comparison": comparison,
    }
    if output_path is not None:
        path = Path(output_path)
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(
            json.dumps(report, sort_keys=True, indent=2) + "\n",
            encoding="utf-8",
        )
    return report
