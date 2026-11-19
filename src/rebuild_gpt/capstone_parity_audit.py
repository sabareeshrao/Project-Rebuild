"""Auditable forward/generation parity against a pinned, trusted upstream file."""

from hashlib import sha256
import json
from pathlib import Path
import platform
import re

import torch

from rebuild_gpt.capstone_parity import (
    compare_capstone_forward,
    compare_capstone_generation,
)


def _digest(path: str | Path) -> str:
    return sha256(Path(path).read_bytes()).hexdigest()


def run_capstone_parity_audit(
    checkpoint_path: str | Path,
    corpus_path: str | Path,
    upstream_model_path: str | Path,
    *,
    output_path: str | Path | None = None,
    expected_upstream_sha256: str | None = None,
    seed: int = 29,
    batch_size: int = 2,
    prompt: str = "a",
    new_tokens: int = 8,
    temperature: float = 0.8,
    top_k: int | None = 3,
) -> dict:
    """Write 'passed' ONLY after both comparisons genuinely succeed.

    This executes a user-provided Python model.py; review the file and
    optionally require its exact SHA-256 BEFORE any import occurs.
    """
    source_hash = _digest(upstream_model_path)
    if expected_upstream_sha256 is not None:
        if not isinstance(expected_upstream_sha256, str) or not re.fullmatch(
            r"[a-fA-F0-9]{64}", expected_upstream_sha256
        ):
            raise ValueError("expected_upstream_sha256 must be 64 hexadecimal digits")
        if source_hash != expected_upstream_sha256.lower():
            raise ValueError("Trusted upstream model SHA-256 mismatch")

    checkpoint_hash = _digest(checkpoint_path)
    corpus_hash = _digest(corpus_path)
    forward = compare_capstone_forward(
        checkpoint_path, corpus_path, upstream_model_path,
        batch_size=batch_size, seed=seed,
    )
    generation = compare_capstone_generation(
        checkpoint_path, upstream_model_path,
        prompt=prompt, new_tokens=new_tokens, temperature=temperature,
        top_k=top_k, seed=seed,
    )
    report = {
        "format_version": 1,
        "status": "passed",
        "checks": {
            "forward_logits_and_loss": forward,
            "generated_token_ids": generation,
        },
        "sha256": {
            "checkpoint": checkpoint_hash,
            "corpus": corpus_hash,
            "upstream_model_py": source_hash,
        },
        "settings": {
            "batch_size": batch_size,
            "seed": seed,
            "prompt": prompt,
            "new_tokens": new_tokens,
            "temperature": temperature,
            "top_k": top_k,
        },
        "environment": {
            "python": platform.python_version(),
            "torch": str(torch.__version__),
        },
    }
    if output_path is not None:
        destination = Path(output_path)
        destination.parent.mkdir(parents=True, exist_ok=True)
        destination.write_text(
            json.dumps(report, indent=2, sort_keys=True, ensure_ascii=False) + "\n",
            encoding="utf-8",
        )
    return report
