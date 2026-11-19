"""Compare a GPT gradient to a central finite difference in float64.

Autograd computes a derivative by backpropagation. A finite difference
independently approximates the slope by evaluating two nearby weights.
"""

import math

import torch

from rebuild_gpt.gpt import GPTLanguageModel
from rebuild_gpt.model_invariants import _validate_token_batch


def check_parameter_gradient(
    model: GPTLanguageModel,
    tokens: torch.Tensor,
    targets: torch.Tensor,
    *,
    parameter_name: str = "tokens.lookup.weight",
    index: tuple[int, ...] = (0, 0),
    epsilon: float = 1e-5,
    atol: float = 1e-5,
    rtol: float = 1e-3,
) -> dict:
    """Verify ONE analytical gradient against a numerical derivative.

    Requires a float64 CPU model for meaningful numerical accuracy.
    Temporarily changes one weight, restores it in all cases, and
    clears gradients created by the check. The input mode is restored.
    """
    _validate_token_batch(model, tokens)
    if (not isinstance(targets, torch.Tensor)
            or targets.shape != tokens.shape or targets.dtype != torch.long):
        raise ValueError("targets must be torch.long [batch, time]")
    if not bool(((targets == -1) |
                 ((targets >= 0) & (targets < model.config.vocab_size))).all()):
        raise ValueError("Targets must be in vocabulary or -1")
    if not bool((targets != -1).any()):
        raise ValueError("At least one supervised target is required")
    if (isinstance(epsilon, bool) or not isinstance(epsilon, (float, int))
            or not math.isfinite(epsilon) or epsilon <= 0):
        raise ValueError("epsilon must be finite and positive")

    parameters = dict(model.named_parameters())
    if parameter_name not in parameters:
        raise ValueError(f"Unknown named model parameter: {parameter_name}")
    parameter = parameters[parameter_name]
    if parameter.dtype != torch.float64 or parameter.device.type != "cpu":
        raise ValueError("Gradient checking requires CPU float64 model parameters")
    if (not isinstance(index, tuple) or len(index) != parameter.ndim
            or any(type(value) is not int or value < 0 or value >= parameter.size(dim)
                   for dim, value in enumerate(index))):
        raise ValueError("index must identify one scalar in the chosen parameter")
    if not parameter.requires_grad:
        raise ValueError("Chosen parameter must require gradients")
    if tokens.device != parameter.device or targets.device != parameter.device:
        raise ValueError("tokens and targets must be on the model device")

    was_training = model.training
    original = float(parameter[index].detach())
    try:
        model.eval()  # no random dropout during slope measurement
        model.zero_grad(set_to_none=True)
        _, loss = model.forward_nanogpt(tokens, targets)
        if loss is None:
            raise AssertionError("Missing training loss")
        loss.backward()
        if parameter.grad is None:
            raise AssertionError("No autograd gradient for chosen parameter")
        analytic = float(parameter.grad[index])

        with torch.no_grad():
            parameter[index] = original + epsilon
            _, loss_plus = model.forward_nanogpt(tokens, targets)
            parameter[index] = original - epsilon
            _, loss_minus = model.forward_nanogpt(tokens, targets)
        numeric = (float(loss_plus) - float(loss_minus)) / (2.0 * epsilon)
        try:
            torch.testing.assert_close(
                torch.tensor(analytic, dtype=torch.float64),
                torch.tensor(numeric, dtype=torch.float64),
                atol=atol, rtol=rtol,
            )
        except AssertionError as exc:
            raise AssertionError(
                f"Autograd and finite difference mismatch for {parameter_name}{index}"
            ) from exc
        return {
            "parameter": parameter_name,
            "index": list(index),
            "autograd": analytic,
            "finite_difference": numeric,
            "absolute_error": abs(analytic - numeric),
        }
    finally:
        with torch.no_grad():
            parameter[index] = original
        model.zero_grad(set_to_none=True)
        model.train(was_training)
