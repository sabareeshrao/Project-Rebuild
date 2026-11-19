"""Compare a real nanoGPT-style grouped AdamW update at matched weights."""

import math

import torch

from rebuild_gpt.nanogpt_keys import export_nanogpt_state_dict
from rebuild_gpt.optimizer_parity import compare_grouped_optimizer_layout
from rebuild_gpt.optimizers import configure_adamw
from rebuild_gpt.training_parity import compare_gradients_against_upstream


def compare_one_grouped_adamw_step(
    ours, upstream, inputs: torch.Tensor, targets: torch.Tensor,
    *, learning_rate: float = 0.001, weight_decay: float = 0.1,
    atol: float = 2e-5, rtol: float = 1e-5,
) -> dict[str, int | float]:
    """Check gradients, actual optimizer groups and ONE matching update.

    Both models are updated IN MEMORY. Unlike our previous plain-AdamW
    exercise, the upstream model must provide configure_optimizers().
    """
    if (isinstance(learning_rate, bool) or not isinstance(learning_rate, (int, float))
            or not math.isfinite(learning_rate) or learning_rate <= 0):
        raise ValueError("learning_rate must be finite and positive")
    if (isinstance(weight_decay, bool) or not isinstance(weight_decay, (int, float))
            or not math.isfinite(weight_decay) or weight_decay < 0):
        raise ValueError("weight_decay must be finite and nonnegative")
    if not callable(getattr(upstream, "configure_optimizers", None)):
        raise ValueError("Reference must implement configure_optimizers()")

    gradient_result = compare_gradients_against_upstream(
        ours, upstream, inputs, targets, atol=atol, rtol=rtol,
    )
    before = export_nanogpt_state_dict(ours)
    ours_optimizer = configure_adamw(
        ours, learning_rate=learning_rate, weight_decay=weight_decay,
        betas=(0.9, 0.95), device_type="cpu",
    )
    reference_optimizer = upstream.configure_optimizers(
        weight_decay=weight_decay, learning_rate=learning_rate,
        betas=(0.9, 0.95), device_type="cpu",
    )
    groups = compare_grouped_optimizer_layout(
        ours, upstream, ours_optimizer, reference_optimizer,
        weight_decay=weight_decay,
    )
    ours_optimizer.step()
    reference_optimizer.step()

    local = export_nanogpt_state_dict(ours)
    upstream_state = upstream.state_dict()
    if not set(local).issubset(upstream_state):
        raise ValueError("Reference is missing updated parameter names")
    max_error, max_update = 0.0, 0.0
    for key in sorted(local):
        try:
            torch.testing.assert_close(
                local[key], upstream_state[key], atol=atol, rtol=rtol,
            )
        except AssertionError as exc:
            raise AssertionError(f"Updated grouped weight mismatch: {key}: {exc}") from exc
        max_error = max(max_error, (local[key] - upstream_state[key]).abs().max().item())
        max_update = max(max_update, (local[key] - before[key]).abs().max().item())
    return {
        **groups,
        "matched_gradients": gradient_result.matched_tensors,
        "max_gradient_error": gradient_result.max_absolute_error,
        "max_updated_weight_error": max_error,
        "largest_local_parameter_change": max_update,
    }
