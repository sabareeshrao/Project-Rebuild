"""Audit next-token cross-entropy semantics on a trusted saved GPT."""

from hashlib import sha256
import json
from pathlib import Path

import torch

from rebuild_gpt.capstone_parity import prepare_validation_parity_batch
from rebuild_gpt.logit_shift_invariance import check_logit_shift_invariance
from rebuild_gpt.masked_microbatch_loss import check_masked_microbatch_equivalence
from rebuild_gpt.perplexity_reference import compare_uniform_perplexity
from rebuild_gpt.text_checkpoint import load_text_checkpoint
from rebuild_gpt.token_loss_breakdown import inspect_token_losses


def run_next_token_loss_audit(
    checkpoint_path: str | Path,
    corpus_path: str | Path,
    *,
    seed: int = 29,
    batch_size: int = 2,
    output_path: str | Path | None = None,
) -> dict:
    """Recompute four loss invariants from held-out next-token examples.

    Construct intentionally unequal supervised counts per row so the
    weighted-microbatch experiment is meaningful. No GPT weights are
    modified, and no external nanoGPT source is imported.
    """
    if type(seed) is not int or seed < 0:
        raise ValueError("seed must be a nonnegative integer")
    if type(batch_size) is not int or batch_size < 2:
        raise ValueError("batch_size must be an integer of at least two")
    if output_path is not None:
        destination = Path(output_path)
        if destination.resolve() in {
            Path(checkpoint_path).resolve(), Path(corpus_path).resolve()
        }:
            raise ValueError("Audit output must not overwrite checkpoint or corpus")
        destination.unlink(missing_ok=True)

    original_rng = torch.get_rng_state().clone()
    try:
        model, tokenizer = load_text_checkpoint(checkpoint_path)
        if model.config.block_size < 3 or model.config.vocab_size < 2:
            raise ValueError("Loss audit requires block_size>=3 and vocab_size>=2")
        inputs, targets = prepare_validation_parity_batch(
            corpus_path, tokenizer, model.config,
            seed=seed, batch_size=batch_size,
        )
        # Make the first row contribute exactly one valid position,
        # while later rows contribute block_size-1 positions each.
        targets = targets.clone()
        targets[0, 1:] = -1
        model.eval()
        with torch.no_grad():
            logits = model(inputs)

        checks = {
            "logit_shift": check_logit_shift_invariance(
                logits, targets, offset=2.0,
            ),
            "uniform_perplexity": compare_uniform_perplexity(logits, targets),
            "token_loss_breakdown": inspect_token_losses(logits, targets),
            "masked_microbatch": check_masked_microbatch_equivalence(
                logits, targets, split_rows=1,
            ),
        }
        counts = checks["masked_microbatch"]["microbatch_supervised_counts"]
        if (checks["logit_shift"]["status"] != "invariant"
                or checks["masked_microbatch"]["status"] != "matched"
                or counts[0] != 1
                or counts[1] <= 1):
            raise AssertionError("Loss conformance checks or uneven masks failed")
        total = checks["uniform_perplexity"]["supervised_positions"]
        if (total != sum(counts)
                or total != checks["token_loss_breakdown"]["supervised_positions"]):
            raise AssertionError("Inconsistent supervised token accounting")
        if abs(checks["token_loss_breakdown"]["mean_nll"]
               - checks["masked_microbatch"]["full_cross_entropy"]) > 2e-6:
            raise AssertionError("Reconstructed loss differs from full-batch CE")

        report = {
            "format_version": 1,
            "status": "passed",
            "scope": "held-out CPU GPT next-token loss and masked accumulation",
            "settings": {"seed": seed, "batch_size": batch_size},
            "sha256": {
                "checkpoint": sha256(Path(checkpoint_path).read_bytes()).hexdigest(),
                "corpus": sha256(Path(corpus_path).read_bytes()).hexdigest(),
            },
            "vocab_size": model.config.vocab_size,
            "block_size": model.config.block_size,
            "checks": checks,
        }
        if output_path is not None:
            destination.parent.mkdir(parents=True, exist_ok=True)
            destination.write_text(
                json.dumps(report, indent=2, sort_keys=True) + "\n",
                encoding="utf-8",
            )
        return report
    finally:
        torch.set_rng_state(original_rng)
