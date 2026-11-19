"""Run four independent GPT semantics checks using one saved model."""

from dataclasses import replace
from hashlib import sha256
import json
from pathlib import Path

import torch

from rebuild_gpt.capstone_parity import prepare_validation_parity_batch
from rebuild_gpt.dropout_semantics import check_dropout_semantics
from rebuild_gpt.gpt import GPTLanguageModel
from rebuild_gpt.masked_gradient_parity import check_masked_target_gradients
from rebuild_gpt.next_token_distribution import inspect_next_token_distribution
from rebuild_gpt.text_checkpoint import load_text_checkpoint
from rebuild_gpt.tied_gradient_decomposition import check_tied_gradient_decomposition


def run_model_semantics(
    checkpoint_path: str | Path,
    corpus_path: str | Path,
    *,
    seed: int = 17,
    output_path: str | Path | None = None,
) -> dict:
    """Audit dropout, ignored gradients, tied gradients and token sampling.

    Existing checkpoints often have dropout=0. To study actual train/eval
    dropout behavior, load the *same weights* into a disposable model
    configured with dropout=0.35. No saved model file is modified.
    """
    if type(seed) is not int or seed < 0:
        raise ValueError("seed must be a nonnegative integer")
    if output_path is not None:
        destination = Path(output_path)
        if destination.resolve() in {
            Path(checkpoint_path).resolve(), Path(corpus_path).resolve(),
        }:
            raise ValueError("Audit output cannot overwrite checkpoint or corpus")
        destination.unlink(missing_ok=True)

    previous_rng = torch.get_rng_state().clone()
    try:
        model, tokenizer = load_text_checkpoint(checkpoint_path)
        if not model.config.tie_embeddings:
            raise ValueError("This semantics lab requires tied GPT embeddings")
        if model.config.block_size < 2 or model.config.vocab_size < 2:
            raise ValueError("Model needs context and vocabulary sizes of at least 2")
        tokens, targets = prepare_validation_parity_batch(
            corpus_path, tokenizer, model.config, batch_size=2, seed=seed,
        )
        masked_targets = targets.clone()
        masked_targets[:, -1] = -1

        dropout_model = GPTLanguageModel(
            replace(model.config, dropout=0.35)
        )
        dropout_model.load_state_dict(model.state_dict(), strict=True)
        dropout = check_dropout_semantics(dropout_model, tokens, seed=seed)
        ignored = check_masked_target_gradients(model, tokens, masked_targets)
        tying = check_tied_gradient_decomposition(model, tokens, targets)

        ids = tokenizer.encode(Path(corpus_path).read_text(encoding="utf-8"))
        if len(ids) < model.config.block_size + 2:
            raise ValueError("Corpus is too short for a long sampling prompt")
        prompt = torch.tensor(
            [ids[:model.config.block_size + 2]], dtype=torch.long,
        )
        distribution = inspect_next_token_distribution(
            model, prompt, temperature=0.8, top_k=3,
        )
        if (dropout.get("status") != "passed"
                or ignored.get("status") != "matched"
                or tying.get("status") != "matched"
                or not distribution.get("allowed_ids")):
            raise AssertionError("A GPT semantics invariant did not pass")
        report = {
            "format_version": 1,
            "status": "passed",
            "scope": "CPU GPT dropout, masked gradients, tied gradients and sampling",
            "settings": {"seed": seed, "dropout_test_probability": 0.35},
            "sha256": {
                "checkpoint": sha256(Path(checkpoint_path).read_bytes()).hexdigest(),
                "corpus": sha256(Path(corpus_path).read_bytes()).hexdigest(),
            },
            "checks": {
                "dropout": dropout,
                "ignored_target_gradients": ignored,
                "tied_weight_gradients": tying,
                "next_token_distribution": distribution,
            },
        }
        if output_path is not None:
            destination.parent.mkdir(parents=True, exist_ok=True)
            destination.write_text(
                json.dumps(report, sort_keys=True, indent=2) + "\n",
                encoding="utf-8",
            )
        return report
    finally:
        torch.set_rng_state(previous_rng)
