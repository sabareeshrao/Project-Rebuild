"""Show how a shared GPT embedding receives two gradient contributions."""

from copy import deepcopy

import torch
from torch import nn

from rebuild_gpt.embedding_tie_check import check_embedding_tying
from rebuild_gpt.gpt import GPTLanguageModel
from rebuild_gpt.model_invariants import _validate_token_batch


def check_tied_gradient_decomposition(
    model: GPTLanguageModel,
    tokens: torch.Tensor,
    targets: torch.Tensor,
    *,
    atol: float = 2e-5,
    rtol: float = 1e-5,
) -> dict:
    """A tied weight's gradient equals input + output untied gradients.

    Identical weights feed two disposable model copies; only one copy
    keeps parameter sharing. Both use eval mode to disable dropout.
    """
    check_embedding_tying(model)
    _validate_token_batch(model, tokens)
    if (not isinstance(targets, torch.Tensor)
            or targets.shape != tokens.shape or targets.dtype != torch.long):
        raise ValueError("targets must be torch.long [batch, time]")
    if not bool(((targets >= 0) & (targets < model.config.vocab_size)).all()):
        raise ValueError("Use real vocabulary target IDs for this experiment")
    if tokens.device.type != "cpu" or next(model.parameters()).device.type != "cpu":
        raise ValueError("Tied-weight gradient decomposition currently requires CPU")

    shared = deepcopy(model).eval()
    separated = deepcopy(model).eval()
    separated.head.linear.weight = nn.Parameter(
        separated.head.linear.weight.detach().clone()
    )
    shared.zero_grad(set_to_none=True)
    separated.zero_grad(set_to_none=True)

    logits_tied, loss_tied = shared.forward_nanogpt(tokens, targets)
    logits_untied, loss_untied = separated.forward_nanogpt(tokens, targets)
    try:
        torch.testing.assert_close(logits_tied, logits_untied, atol=atol, rtol=rtol)
        torch.testing.assert_close(loss_tied, loss_untied, atol=atol, rtol=rtol)
    except AssertionError as exc:
        raise AssertionError("Tied and untied models began with different outputs") from exc
    loss_tied.backward()
    loss_untied.backward()

    tied_grad = shared.tokens.lookup.weight.grad
    input_grad = separated.tokens.lookup.weight.grad
    output_grad = separated.head.linear.weight.grad
    if tied_grad is None or input_grad is None or output_grad is None:
        raise AssertionError("Missing gradient contribution from the shared weight")
    expected = input_grad + output_grad
    try:
        torch.testing.assert_close(tied_grad, expected, atol=atol, rtol=rtol)
    except AssertionError as exc:
        raise AssertionError("Tied weight gradient does not equal both contributions") from exc

    shared_params = dict(shared.named_parameters())
    untied_params = dict(separated.named_parameters())
    for name, parameter in shared_params.items():
        if name == "tokens.lookup.weight":
            continue
        match = untied_params.get(name)
        if match is None or parameter.grad is None or match.grad is None:
            raise AssertionError(f"Missing reference gradient for {name}")
        try:
            torch.testing.assert_close(parameter.grad, match.grad,
                                       atol=atol, rtol=rtol)
        except AssertionError as exc:
            raise AssertionError(f"Non-shared gradient changed at {name}") from exc

    return {
        "status": "matched",
        "shared_gradient_error": float((tied_grad - expected).abs().max()),
        "input_contribution_norm": float(input_grad.norm()),
        "output_contribution_norm": float(output_grad.norm()),
        "checked_other_parameters": len(shared_params) - 1,
    }
