"""Compare a real nanoGPT-style grouped AdamW update at matched weights."""

from collections.abc import Sequence
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


def compare_grouped_adamw_trajectory(
    ours, upstream, inputs: torch.Tensor, targets: torch.Tensor,
    *, steps: int = 3, learning_rate: float = 0.001, weight_decay: float = 0.1,
    atol: float = 2e-5, rtol: float = 1e-5,
    batches: Sequence[tuple[torch.Tensor, torch.Tensor]] | None = None,
) -> dict:
    """Compare repeated steps with PERSISTENT AdamW optimizer state.

    No model re-synchronization, optimizer re-creation or reseeding
    occurs between steps. Both models run with dropout disabled.
    """
    from rebuild_gpt.training_parity import capture_upstream_named_gradients
    from rebuild_gpt.upstream_parity import synchronize_upstream_weights

    if type(steps) is not int or steps < 1:
        raise ValueError("steps must be a positive integer")
    if (isinstance(learning_rate, bool) or not isinstance(learning_rate, (int, float))
            or not math.isfinite(learning_rate) or learning_rate <= 0):
        raise ValueError("learning_rate must be finite and positive")
    if (isinstance(weight_decay, bool) or not isinstance(weight_decay, (int, float))
            or not math.isfinite(weight_decay) or weight_decay < 0):
        raise ValueError("weight_decay must be finite and nonnegative")
    if (inputs.ndim != 2 or inputs.numel() == 0 or inputs.dtype != torch.long
            or targets.shape != inputs.shape or targets.dtype != torch.long):
        raise ValueError("inputs and targets must be non-empty [B,T] long tensors")
    if not callable(getattr(upstream, "configure_optimizers", None)):
        raise ValueError("Reference must implement configure_optimizers()")
    if batches is None:
        step_pairs = tuple((inputs, targets) for _ in range(steps))
    else:
        if not isinstance(batches, (tuple, list)) or len(batches) != steps:
            raise ValueError("batches must have exactly one pair per step")
        step_pairs = tuple(batches)
        for item in step_pairs:
            if not isinstance(item, (tuple, list)) or len(item) != 2:
                raise ValueError("Each batch must be an (inputs, targets) pair")
            x, y = item
            if (not isinstance(x, torch.Tensor) or not isinstance(y, torch.Tensor)
                    or x.ndim != 2 or x.numel() == 0 or x.dtype != torch.long
                    or y.shape != x.shape or y.dtype != torch.long):
                raise ValueError("Each batch pair must be non-empty [B,T] long tensors")
            if x.shape != inputs.shape:
                raise ValueError("Every batch must match the input shape")

    synchronize_upstream_weights(ours, upstream)  # Once, before step one.
    local_opt = configure_adamw(
        ours, learning_rate=learning_rate, weight_decay=weight_decay,
        betas=(0.9, 0.95), device_type="cpu",
    )
    reference_opt = upstream.configure_optimizers(
        weight_decay=weight_decay, learning_rate=learning_rate,
        betas=(0.9, 0.95), device_type="cpu",
    )
    groups = compare_grouped_optimizer_layout(
        ours, upstream, local_opt, reference_opt, weight_decay=weight_decay,
    )
    previous_local_mode, previous_reference_mode = ours.training, upstream.training
    history = []
    try:
        ours.eval()
        upstream.eval()
        for step, (step_inputs, step_targets) in enumerate(step_pairs, start=1):
            local_opt.zero_grad(set_to_none=True)
            reference_opt.zero_grad(set_to_none=True)
            _, local_loss = ours.forward_nanogpt(step_inputs, step_targets)
            _, reference_loss = upstream(step_inputs, step_targets)
            torch.testing.assert_close(
                local_loss.detach(), reference_loss.detach(),
                atol=atol, rtol=rtol,
            )
            local_loss.backward()
            reference_loss.backward()

            local_grads = capture_upstream_named_gradients(ours)
            upstream_grads = {
                name: param.grad.detach().clone()
                for name, param in upstream.named_parameters()
                if param.grad is not None
            }
            if set(local_grads) != set(upstream_grads):
                raise ValueError("Backward gradient keys differ")
            gradient_error = 0.0
            for name in sorted(local_grads):
                try:
                    torch.testing.assert_close(
                        local_grads[name], upstream_grads[name],
                        atol=atol, rtol=rtol,
                    )
                except AssertionError as exc:
                    raise AssertionError(
                        f"Step {step} gradient mismatch at {name}: {exc}"
                    ) from exc
                gradient_error = max(
                    gradient_error,
                    (local_grads[name] - upstream_grads[name]).abs().max().item(),
                )
            local_opt.step()
            reference_opt.step()
            local_state = export_nanogpt_state_dict(ours)
            reference_state = upstream.state_dict()
            if not set(local_state).issubset(reference_state):
                raise ValueError("Missing upstream state keys after optimizer update")
            weight_error = 0.0
            for name in sorted(local_state):
                try:
                    torch.testing.assert_close(
                        local_state[name], reference_state[name],
                        atol=atol, rtol=rtol,
                    )
                except AssertionError as exc:
                    raise AssertionError(
                        f"Step {step} updated parameter mismatch at {name}: {exc}"
                    ) from exc
                weight_error = max(
                    weight_error,
                    (local_state[name] - reference_state[name]).abs().max().item(),
                )
            history.append({
                "step": step, "loss": float(local_loss.detach()),
                "max_gradient_error": gradient_error,
                "max_updated_weight_error": weight_error,
            })
    finally:
        ours.train(previous_local_mode)
        upstream.train(previous_reference_mode)
    return {"steps": steps, "group_layout": groups, "history": history}
