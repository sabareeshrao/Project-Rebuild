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


def check_batch_independence(
    model: GPTLanguageModel,
    tokens: torch.Tensor,
    *,
    atol: float = 2e-5,
    rtol: float = 1e-5,
) -> dict[str, int | float]:
    """A sample's logits must not change when other batch rows are present.

    Eval mode removes per-call random dropout. This detects accidental
    cross-example attention, normalization or batch-index dependence.
    """
    _validate_token_batch(model, tokens, min_batch=2)
    original_mode = model.training
    try:
        model.eval()
        with torch.no_grad():
            together = model(tokens)
            separately = torch.cat(
                [model(tokens[row:row + 1]) for row in range(tokens.size(0))],
                dim=0,
            )
        try:
            torch.testing.assert_close(together, separately, atol=atol, rtol=rtol)
        except AssertionError as exc:
            raise AssertionError(
                "Batch independence violated: other examples changed a sample's logits"
            ) from exc
        return {
            "checked_samples": tokens.size(0),
            "max_batch_error": float((together - separately).abs().max()),
        }
    finally:
        model.train(original_mode)


def check_forward_contract(
    model: GPTLanguageModel,
    tokens: torch.Tensor,
    targets: torch.Tensor,
    *,
    atol: float = 2e-5,
    rtol: float = 1e-5,
) -> dict[str, int | float]:
    """Validate nanoGPT's training logits, inference logits and masked loss.

    With targets: [B,T,V] logits and cross-entropy loss ignoring -1.
    Without targets: [B,1,V] final-position logits and None loss.
    """
    from torch.nn import functional as F

    _validate_token_batch(model, tokens)
    if (not isinstance(targets, torch.Tensor)
            or targets.shape != tokens.shape or targets.dtype != torch.long):
        raise ValueError("targets must be torch.long with shape [batch, time]")
    valid_targets = (targets == -1) | ((targets >= 0) & (targets < model.config.vocab_size))
    if not bool(valid_targets.all()):
        raise ValueError("Targets must be vocabulary IDs or -1")
    supervised = int((targets != -1).sum())
    if supervised == 0:
        raise ValueError("At least one supervised token is required")

    original_mode = model.training
    try:
        model.eval()
        with torch.no_grad():
            full = model(tokens)
            train_logits, loss = model.forward_nanogpt(tokens, targets)
            final_logits, no_loss = model.forward_nanogpt(tokens)
            expected_loss = F.cross_entropy(
                full.reshape(-1, full.size(-1)),
                targets.reshape(-1),
                ignore_index=-1,
            )
        if loss is None or no_loss is not None:
            raise AssertionError("nanoGPT forward returned an incorrect loss contract")
        try:
            torch.testing.assert_close(train_logits, full, atol=atol, rtol=rtol)
            torch.testing.assert_close(final_logits, full[:, -1:, :],
                                       atol=atol, rtol=rtol)
            torch.testing.assert_close(loss, expected_loss, atol=atol, rtol=rtol)
        except AssertionError as exc:
            raise AssertionError(
                "nanoGPT forward contract mismatch: logits, final position or masked loss"
            ) from exc
        return {
            "supervised_positions": supervised,
            "ignored_positions": targets.numel() - supervised,
            "max_train_logit_error": float((train_logits - full).abs().max()),
            "max_last_logit_error": float(
                (final_logits - full[:, -1:, :]).abs().max()
            ),
            "masked_loss_error": float((loss - expected_loss).abs()),
        }
    finally:
        model.train(original_mode)
