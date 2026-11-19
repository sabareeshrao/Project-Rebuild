"""Verify that ignored next-token labels affect neither GPT loss nor gradients."""

from copy import deepcopy

import torch
from torch.nn import functional as F

from rebuild_gpt.gpt import GPTLanguageModel
from rebuild_gpt.model_invariants import _validate_token_batch


def check_masked_target_gradients(
    model: GPTLanguageModel,
    tokens: torch.Tensor,
    targets: torch.Tensor,
    *,
    atol: float = 2e-5,
    rtol: float = 1e-5,
) -> dict:
    """Compare ignore_index=-1 backward pass to selecting valid labels.

    The independent calculation selects only supervised logits before
    computing cross-entropy. Separate model copies avoid touching the
    original model's weights, gradients, or current train/eval mode.
    """
    _validate_token_batch(model, tokens)
    if (not isinstance(targets, torch.Tensor)
            or targets.shape != tokens.shape or targets.dtype != torch.long):
        raise ValueError("targets must be torch.long [batch, time]")
    valid = (targets == -1) | ((targets >= 0) & (targets < model.config.vocab_size))
    if not bool(valid.all()):
        raise ValueError("Target IDs must be vocabulary tokens or -1")
    selected = targets != -1
    if not bool(selected.any()) or bool(selected.all()):
        raise ValueError("Need both supervised and ignored target positions")
    if tokens.device.type != "cpu" or next(model.parameters()).device.type != "cpu":
        raise ValueError("Masked gradient comparison currently requires CPU")

    with_ignore = deepcopy(model).eval()
    manual_selection = deepcopy(model).eval()
    with_ignore.zero_grad(set_to_none=True)
    manual_selection.zero_grad(set_to_none=True)
    logits, loss = with_ignore.forward_nanogpt(tokens, targets)
    if loss is None:
        raise AssertionError("GPT ignored-target forward returned no loss")
    loss.backward()
    manual_logits = manual_selection(tokens)
    expected_loss = F.cross_entropy(manual_logits[selected], targets[selected])
    expected_loss.backward()

    try:
        torch.testing.assert_close(loss.detach(), expected_loss.detach(),
                                   atol=atol, rtol=rtol)
    except AssertionError as exc:
        raise AssertionError("Ignored-target GPT loss differs from selected-token CE") from exc
    left = dict(with_ignore.named_parameters())
    right = dict(manual_selection.named_parameters())
    if set(left) != set(right):
        raise AssertionError("Gradient parameter keys differ")
    maximum = 0.0
    for name in sorted(left):
        a, b = left[name].grad, right[name].grad
        if a is None or b is None:
            raise AssertionError(f"Missing masked gradient for {name}")
        try:
            torch.testing.assert_close(a, b, atol=atol, rtol=rtol)
        except AssertionError as exc:
            raise AssertionError(f"Ignored-target gradient mismatch for {name}") from exc
        maximum = max(maximum, float((a - b).detach().abs().max()))
    return {
        "status": "matched",
        "supervised_positions": int(selected.sum()),
        "ignored_positions": int((~selected).sum()),
        "matched_parameter_gradients": len(left),
        "max_gradient_error": maximum,
        "loss_error": float((loss - expected_loss).detach().abs()),
    }
