"""Audit numerical accuracy and long-context generation after the milestone."""

from copy import deepcopy
from hashlib import sha256
import json
from pathlib import Path

import torch

from rebuild_gpt.capstone_parity import prepare_validation_parity_batch
from rebuild_gpt.conformance_budgets import check_conformance_budgets
from rebuild_gpt.conformance_integrity import verify_conformance_integrity
from rebuild_gpt.generation_context_check import check_generation_context
from rebuild_gpt.gradient_coordinate_sweep import check_gradient_coordinates
from rebuild_gpt.text_checkpoint import load_text_checkpoint


def run_post_milestone_audit(
    checkpoint_path: str | Path,
    corpus_path: str | Path,
    conformance_report_path: str | Path,
    *,
    output_path: str | Path | None = None,
    seed: int = 37,
    new_tokens: int = 4,
) -> dict:
    """Require independently verified inputs and all new checks to pass.

    This is a CPU-only experiment. The checkpoint must be a trusted local
    model file. Existing conformance and capstone artifacts are read-only.
    """
    if type(seed) is not int or seed < 0:
        raise ValueError("seed must be a nonnegative integer")
    if type(new_tokens) is not int or new_tokens < 1:
        raise ValueError("new_tokens must be a positive integer")
    if output_path is not None:
        destination = Path(output_path)
        if destination.resolve() in {
            Path(checkpoint_path).resolve(),
            Path(corpus_path).resolve(),
            Path(conformance_report_path).resolve(),
        }:
            raise ValueError("Audit output must not overwrite its input artifacts")
        destination.unlink(missing_ok=True)

    integrity = verify_conformance_integrity(
        conformance_report_path, checkpoint_path, corpus_path,
    )
    budgets = check_conformance_budgets(conformance_report_path)
    model, tokenizer = load_text_checkpoint(checkpoint_path)
    if model.config.vocab_size < 2 or model.config.n_embd < 2:
        raise ValueError("The post-milestone audit needs vocab_size and n_embd >= 2")
    tokens, targets = prepare_validation_parity_batch(
        corpus_path, tokenizer, model.config, batch_size=2, seed=seed,
    )
    gradients = check_gradient_coordinates(deepcopy(model).double(), tokens, targets)

    corpus_ids = tokenizer.encode(Path(corpus_path).read_text(encoding="utf-8"))
    long_length = model.config.block_size + 2
    if len(corpus_ids) < long_length:
        raise ValueError("Corpus too short to build an overlong prompt")
    long_prompt = torch.tensor(
        [corpus_ids[:long_length]], dtype=torch.long,
    )
    context = check_generation_context(
        model, long_prompt, new_tokens=new_tokens, seed=seed,
    )
    if (integrity.get("status") != "verified"
            or budgets.get("status") != "within_budget"
            or gradients.get("status") != "passed"
            or context.get("status") != "matched"):
        raise AssertionError("Post-milestone correctness check failed")
    result = {
        "format_version": 1,
        "status": "passed",
        "scope": "CPU GPT error budgets, multiple finite differences and context cropping",
        "sha256": {
            "checkpoint": integrity["checkpoint_sha256"],
            "corpus": integrity["corpus_sha256"],
            "source_conformance_report": sha256(
                Path(conformance_report_path).read_bytes()
            ).hexdigest(),
        },
        "settings": {"seed": seed, "new_tokens": new_tokens},
        "numerical_budgets": budgets,
        "gradient_coordinates": gradients,
        "long_prompt_generation": context,
    }
    if output_path is not None:
        destination.parent.mkdir(parents=True, exist_ok=True)
        destination.write_text(
            json.dumps(result, indent=2, sort_keys=True) + "\n",
            encoding="utf-8",
        )
    return result
