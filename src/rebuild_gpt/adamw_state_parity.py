"""Inspect AdamW's persistent first/second moments by parameter name."""

import torch

from rebuild_gpt.nanogpt_keys import upstream_key


def named_adamw_moments(model, optimizer) -> dict[str, dict[str, torch.Tensor]]:
    """Snapshot AdamW moments for each *unique* trainable model parameter.

    Reject uninitialized/missing optimizer state. A state snapshot is only
    meaningful after at least one optimizer step with all gradients present.
    """
    parameters = {id(param): (name, param) for name, param in model.named_parameters()
                  if param.requires_grad}
    result: dict[str, dict[str, torch.Tensor]] = {}
    seen: set[int] = set()
    for group in optimizer.param_groups:
        for param in group["params"]:
            identity = id(param)
            if identity not in parameters:
                raise ValueError("Optimizer contains parameter outside model")
            if identity in seen:
                raise ValueError("Optimizer repeats a trainable parameter")
            seen.add(identity)
            name, _ = parameters[identity]
            raw = optimizer.state.get(param, {})
            for key in ("step", "exp_avg", "exp_avg_sq"):
                if key not in raw or not isinstance(raw[key], torch.Tensor):
                    raise ValueError(f"Missing AdamW moment {key} for {name}")
            result[name] = {
                key: raw[key].detach().clone()
                for key in ("step", "exp_avg", "exp_avg_sq")
            }
    if seen != set(parameters):
        raise ValueError("Optimizer is missing trainable model parameters")
    return result


def compare_adamw_moments(
    ours, upstream, local_optimizer, upstream_optimizer,
    *, atol: float = 2e-5, rtol: float = 1e-5,
) -> dict[str, int | float]:
    """Compare AdamW step counters, first and second moments by names."""
    local = {
        upstream_key(name): values
        for name, values in named_adamw_moments(ours, local_optimizer).items()
    }
    reference = named_adamw_moments(upstream, upstream_optimizer)
    if set(local) != set(reference):
        raise ValueError("AdamW state parameter name mismatch")

    errors = {"step": 0.0, "exp_avg": 0.0, "exp_avg_sq": 0.0}
    for name in sorted(local):
        for key in errors:
            a, b = local[name][key], reference[name][key]
            try:
                torch.testing.assert_close(a, b, atol=atol, rtol=rtol)
            except AssertionError as exc:
                raise AssertionError(f"AdamW {key} mismatch for {name}: {exc}") from exc
            errors[key] = max(errors[key], (a - b).abs().max().item())
    return {
        "matched_optimizer_states": len(local),
        "max_step_error": errors["step"],
        "max_exp_avg_error": errors["exp_avg"],
        "max_exp_avg_sq_error": errors["exp_avg_sq"],
    }
