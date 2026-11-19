"""Verify why GPT dropout differs in train and evaluation modes."""

import torch

from rebuild_gpt.gpt import GPTLanguageModel
from rebuild_gpt.model_invariants import _validate_token_batch


def check_dropout_semantics(
    model: GPTLanguageModel,
    tokens: torch.Tensor,
    *,
    seed: int = 17,
) -> dict:
    """Eval predictions repeat; training predictions should vary with masks.

    Use a fixed private experiment seed and restore both process RNG and
    the model's previous train/eval mode in all outcomes. This is an
    educational stochastic check on a nontrivial token batch.
    """
    _validate_token_batch(model, tokens)
    if not 0.0 < model.config.dropout < 1.0:
        raise ValueError("Dropout semantics need a probability between zero and one")
    if type(seed) is not int or seed < 0:
        raise ValueError("seed must be a nonnegative integer")
    if tokens.device.type != "cpu" or next(model.parameters()).device.type != "cpu":
        raise ValueError("Dropout semantics check currently requires CPU")
    previous_mode = model.training
    previous_rng = torch.get_rng_state().clone()
    try:
        torch.manual_seed(seed)
        with torch.no_grad():
            model.eval()
            eval_first = model(tokens)
            eval_second = model(tokens)
            model.train()
            train_first = model(tokens)
            train_second = model(tokens)
        try:
            torch.testing.assert_close(eval_first, eval_second, atol=0, rtol=0)
        except AssertionError as exc:
            raise AssertionError("Dropout changed predictions in eval mode") from exc
        train_difference = float((train_first - train_second).abs().max())
        if train_difference == 0.0:
            raise AssertionError("Dropout training mode produced identical predictions")
        return {
            "status": "passed",
            "seed": seed,
            "evaluated_positions": tokens.numel(),
            "eval_repeat_error": float((eval_first - eval_second).abs().max()),
            "training_output_difference": train_difference,
            "rng_preserved": True,
        }
    finally:
        torch.set_rng_state(previous_rng)
        model.train(previous_mode)
