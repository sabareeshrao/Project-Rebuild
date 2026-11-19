"""Run independent GPT behavioral checks against a saved character model."""

from copy import deepcopy
from hashlib import sha256
import json
from pathlib import Path
import platform

import torch

from rebuild_gpt.capstone_parity import prepare_validation_parity_batch
from rebuild_gpt.gradient_check import check_parameter_gradient
from rebuild_gpt.model_invariants import (
    check_batch_independence,
    check_causal_prefix,
    check_forward_contract,
)
from rebuild_gpt.text_checkpoint import load_text_checkpoint


def run_model_conformance(
    checkpoint_path: str | Path,
    corpus_path: str | Path,
    *,
    batch_size: int = 2,
    seed: int = 29,
    output_path: str | Path | None = None,
) -> dict:
    """Run four checks; write 'passed' only if all truly succeed.

    No upstream Python file is imported. The gradient check is performed
    on a float64 COPY of the model so the loaded weights remain untouched.
    The corpus/tokenizer and batch seed must match the saved checkpoint.
    """
    if type(batch_size) is not int or batch_size < 2:
        raise ValueError("batch_size must be an integer of at least two")
    if type(seed) is not int or seed < 0:
        raise ValueError("seed must be a nonnegative integer")
    if output_path is not None:
        Path(output_path).unlink(missing_ok=True)

    model, tokenizer = load_text_checkpoint(checkpoint_path)
    if model.config.block_size < 2 or model.config.vocab_size < 2:
        raise ValueError("Conformance needs block_size and vocab_size >= 2")
    tokens, targets = prepare_validation_parity_batch(
        corpus_path, tokenizer, model.config,
        batch_size=batch_size, seed=seed,
    )
    checks = {
        "causal_prefix": check_causal_prefix(model, tokens),
        "batch_independence": check_batch_independence(model, tokens),
        "forward_contract": check_forward_contract(model, tokens, targets),
        "finite_difference": check_parameter_gradient(
            deepcopy(model).double(), tokens, targets,
        ),
    }
    report = {
        "format_version": 1,
        "status": "passed",
        "scope": "saved GPT behavior and one double-precision derivative on held-out tokens",
        "checks": checks,
        "settings": {"batch_size": batch_size, "seed": seed},
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
        "environment": {
            "python": platform.python_version(),
            "torch": str(torch.__version__),
        },
    }
    if output_path is not None:
        destination = Path(output_path)
        destination.parent.mkdir(parents=True, exist_ok=True)
        destination.write_text(
            json.dumps(report, indent=2, sort_keys=True) + "\n",
            encoding="utf-8",
        )
    return report
