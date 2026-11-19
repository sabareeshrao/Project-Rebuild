"""Behavioral invariants that expose mistakes hidden by a low loss.

The tests apply to any supported GPT attention implementation and do
not require importing the external nanoGPT source.
"""

import torch

from rebuild_gpt.gpt import GPTLanguageModel


def _validate_token_batch(model: GPTLanguageModel, tokens: torch.Tensor,
                          *, min_batch: int = 1, min_time: int = 1) -> None:
    if (not isinstance(tokens, torch.Tensor) or tokens.ndim != 2
            or tokens.dtype != torch.long
            or tokens.size(0) < min_batch or tokens.size(1) < min_time):
        raise ValueError("Expected non-empty torch.long tokens [batch, time]")
    if tokens.size(1) > model.config.block_size:
        raise ValueError("Tokens exceed the model's block_size")
    if (tokens < 0).any() or (tokens >= model.config.vocab_size).any():
        raise ValueError("Token ID outside model vocabulary")


def check_causal_prefix(
    model: GPTLanguageModel,
    tokens: torch.Tensor,
    *,
    atol: float = 2e-5,
    rtol: float = 1e-5,
) -> dict[str, int | float]:
    """Changing only the LAST token must not affect earlier-position logits.

    This metamorphic test needs no predetermined expected logits. It uses
    eval mode to disable dropout and restores the incoming training mode.
    """
    _validate_token_batch(model, tokens, min_time=2)
    if model.config.vocab_size < 2:
        raise ValueError("Need at least two vocabulary IDs to change a token")
    original_mode = model.training
    try:
        model.eval()
        changed = tokens.clone()
        changed[:, -1] = (changed[:, -1] + 1) % model.config.vocab_size
        with torch.no_grad():
            baseline = model(tokens)
            perturbed = model(changed)
        earlier = baseline[:, :-1, :]
        alternate = perturbed[:, :-1, :]
        try:
            torch.testing.assert_close(earlier, alternate, atol=atol, rtol=rtol)
        except AssertionError as exc:
            raise AssertionError(
                "Causal invariant violated: future token changed earlier logits"
            ) from exc
        return {
            "checked_positions": tokens.size(0) * (tokens.size(1) - 1),
            "max_prefix_error": float((earlier - alternate).abs().max()),
        }
    finally:
        model.train(original_mode)
