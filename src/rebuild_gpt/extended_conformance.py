"""Build a reproducible extended conformance report for a saved GPT."""

from hashlib import sha256
import json
from pathlib import Path
import platform

import torch

from rebuild_gpt.attention_backend_parity import compare_attention_backends
from rebuild_gpt.capstone_parity import prepare_validation_parity_batch
from rebuild_gpt.checkpoint_roundtrip import check_checkpoint_roundtrip
from rebuild_gpt.embedding_tie_check import check_embedding_tying
from rebuild_gpt.model_conformance import run_model_conformance
from rebuild_gpt.seeded_generation_check import check_seeded_generation
from rebuild_gpt.text_checkpoint import load_text_checkpoint


def run_extended_conformance(
    checkpoint_path: str | Path,
    corpus_path: str | Path,
    *,
    batch_size: int = 2,
    seed: int = 29,
    output_path: str | Path | None = None,
) -> dict:
    """Require the original four checks plus four independent invariants.

    The report is saved only after all eight tests succeed. The two
    model-instantiation checks preserve global RNG; generated tokens use
    a private generator. No upstream Python code is imported.
    """
    if output_path is not None:
        Path(output_path).unlink(missing_ok=True)
    base = run_model_conformance(
        checkpoint_path, corpus_path,
        batch_size=batch_size, seed=seed,
    )
    model, tokenizer = load_text_checkpoint(checkpoint_path)
    if model.config.attention_impl not in ("fused", "sdpa"):
        raise ValueError("Extended conformance requires fused or sdpa attention")
    tokens, targets = prepare_validation_parity_batch(
        corpus_path, tokenizer, model.config,
        batch_size=batch_size, seed=seed,
    )
    extra = {
        "embedding_tying": check_embedding_tying(model),
        "seeded_generation": check_seeded_generation(
            model, tokens[:1, :min(tokens.size(1), 4)],
            new_tokens=4, seed=seed,
        ),
        "checkpoint_roundtrip": check_checkpoint_roundtrip(
            checkpoint_path, tokens,
        ),
        "attention_backends": compare_attention_backends(model, tokens, targets),
    }
    report = {
        "format_version": 1,
        "status": "passed",
        "scope": "eight CPU GPT behavioral and numerical conformance checks",
        "settings": {"seed": seed, "batch_size": batch_size},
        "sha256": {
            "checkpoint": sha256(Path(checkpoint_path).read_bytes()).hexdigest(),
            "corpus": sha256(Path(corpus_path).read_bytes()).hexdigest(),
        },
        "model": {
            "attention_impl": model.config.attention_impl,
            "vocab_size": model.config.vocab_size,
            "block_size": model.config.block_size,
            "n_layer": model.config.n_layer,
        },
        "checks": {**base["checks"], **extra},
        "environment": {
            "python": platform.python_version(),
            "torch": str(torch.__version__),
        },
    }
    if len(report["checks"]) != 8:
        raise AssertionError("Conformance audit must contain exactly eight checks")
    if output_path is not None:
        location = Path(output_path)
        location.parent.mkdir(parents=True, exist_ok=True)
        location.write_text(
            json.dumps(report, indent=2, sort_keys=True) + "\n",
            encoding="utf-8",
        )
    return report
