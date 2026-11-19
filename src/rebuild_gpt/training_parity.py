"""Gradient snapshots for comparing our trained GPT to a trusted reference."""

import torch

from rebuild_gpt.gpt import GPTLanguageModel
from rebuild_gpt.nanogpt_keys import upstream_key


def capture_upstream_named_gradients(model: GPTLanguageModel) -> dict[str, torch.Tensor]:
    """Clone one gradient per unique parameter, using nanoGPT-style names.

    Call loss.backward() first. PyTorch deduplicates tied embedding/head
    parameters in named_parameters(), so tied weights are counted once.
    """
    if model.config.attention_impl not in ("fused", "sdpa") or not model.config.tie_embeddings:
        raise ValueError("Gradient capture requires fused/SDPA attention and tied embeddings")
    result: dict[str, torch.Tensor] = {}
    for local_name, parameter in model.named_parameters():
        if parameter.grad is None:
            raise ValueError(f"Missing gradient for {local_name}; call backward() first")
        key = upstream_key(local_name)
        if key in result:
            raise ValueError(f"Duplicate mapped gradient name: {key}")
        result[key] = parameter.grad.detach().clone()
    return result


from dataclasses import dataclass

from rebuild_gpt.upstream_parity import synchronize_upstream_weights


@dataclass(frozen=True)
class GradientParityResult:
    matched_tensors: int
    max_absolute_error: float


def compare_gradients_against_upstream(
    ours: GPTLanguageModel,
    upstream: torch.nn.Module,
    inputs: torch.Tensor,
    targets: torch.Tensor,
    *,
    atol: float = 2e-5,
    rtol: float = 1e-5,
) -> GradientParityResult:
    """Synchronize weights, then compare matched-weight backward gradients.

    This clears and replaces gradients on BOTH models, but restores their
    previous train/eval modes. No optimizer step is performed here.
    """
    if (inputs.ndim != 2 or inputs.numel() == 0 or inputs.dtype != torch.long
            or targets.shape != inputs.shape or targets.dtype != torch.long):
        raise ValueError("inputs and targets must be non-empty [B,T] long tensors")

    synchronize_upstream_weights(ours, upstream)
    ours_was_training, upstream_was_training = ours.training, upstream.training
    try:
        ours.eval()
        upstream.eval()
        ours.zero_grad(set_to_none=True)
        upstream.zero_grad(set_to_none=True)
        _, our_loss = ours.forward_nanogpt(inputs, targets)
        _, upstream_loss = upstream(inputs, targets)
        torch.testing.assert_close(
            our_loss.detach(), upstream_loss.detach(), atol=atol, rtol=rtol,
        )
        our_loss.backward()
        upstream_loss.backward()
        local = capture_upstream_named_gradients(ours)
        reference = {
            name: parameter.grad.detach().clone()
            for name, parameter in upstream.named_parameters()
            if parameter.grad is not None
        }
        if set(local) != set(reference):
            raise ValueError(
                "Gradient key mismatch: missing="
                f"{sorted(set(local)-set(reference))}, "
                f"extra={sorted(set(reference)-set(local))}"
            )
        maximum = 0.0
        for name in sorted(local):
            try:
                torch.testing.assert_close(
                    local[name], reference[name], atol=atol, rtol=rtol,
                )
            except AssertionError as exc:
                raise AssertionError(f"Gradient mismatch at {name}: {exc}") from exc
            maximum = max(maximum, (local[name] - reference[name]).abs().max().item())
        return GradientParityResult(len(local), maximum)
    finally:
        ours.train(ours_was_training)
        upstream.train(upstream_was_training)


@dataclass(frozen=True)
class AdamWStepParityResult:
    matched_gradients: int
    max_gradient_error: float
    matched_parameter_tensors: int
    max_updated_weight_error: float
    largest_local_parameter_change: float


def compare_one_adamw_step(
    ours: GPTLanguageModel,
    upstream: torch.nn.Module,
    inputs: torch.Tensor,
    targets: torch.Tensor,
    *,
    learning_rate: float = 0.001,
    weight_decay: float = 0.01,
    atol: float = 2e-5,
    rtol: float = 1e-5,
) -> AdamWStepParityResult:
    """Compare ONE identical AdamW update, not upstream training-loop behavior.

    Both model objects are updated IN PLACE. We first check gradients,
    then apply the same plain torch.optim.AdamW settings to both models.
    This does NOT exercise nanoGPT's grouped configure_optimizers method.
    """
    import math

    from rebuild_gpt.nanogpt_keys import export_nanogpt_state_dict

    if (isinstance(learning_rate, bool) or not isinstance(learning_rate, (float, int))
            or not math.isfinite(learning_rate) or learning_rate <= 0):
        raise ValueError("learning_rate must be finite and positive")
    if (isinstance(weight_decay, bool) or not isinstance(weight_decay, (float, int))
            or not math.isfinite(weight_decay) or weight_decay < 0):
        raise ValueError("weight_decay must be finite and nonnegative")

    gradients = compare_gradients_against_upstream(
        ours, upstream, inputs, targets, atol=atol, rtol=rtol,
    )
    before = export_nanogpt_state_dict(ours)
    # These are deliberately identical PyTorch optimizers on both models.
    local_optimizer = torch.optim.AdamW(
        ours.parameters(), lr=learning_rate, weight_decay=weight_decay,
        betas=(0.9, 0.95), eps=1e-8,
    )
    upstream_optimizer = torch.optim.AdamW(
        upstream.parameters(), lr=learning_rate, weight_decay=weight_decay,
        betas=(0.9, 0.95), eps=1e-8,
    )
    local_optimizer.step()
    upstream_optimizer.step()

    local_after = export_nanogpt_state_dict(ours)
    reference_after = upstream.state_dict()
    missing = set(local_after) - set(reference_after)
    if missing:
        raise ValueError(f"Upstream is missing updated tensor keys: {sorted(missing)}")
    maximum_error = 0.0
    maximum_change = 0.0
    for name in sorted(local_after):
        ours_tensor = local_after[name]
        theirs_tensor = reference_after[name]
        try:
            torch.testing.assert_close(
                ours_tensor, theirs_tensor, atol=atol, rtol=rtol,
            )
        except AssertionError as exc:
            raise AssertionError(f"Updated weight mismatch at {name}: {exc}") from exc
        maximum_error = max(maximum_error, (ours_tensor - theirs_tensor).abs().max().item())
        maximum_change = max(maximum_change, (ours_tensor - before[name]).abs().max().item())
    return AdamWStepParityResult(
        gradients.matched_tensors, gradients.max_absolute_error,
        len(local_after), maximum_error, maximum_change,
    )
