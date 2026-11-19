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
