"""Compare manual fused causal attention with PyTorch SDPA in one GPT."""

from copy import deepcopy
from dataclasses import replace

import torch

from rebuild_gpt.config import GPTConfig
from rebuild_gpt.gpt import GPTLanguageModel
from rebuild_gpt.model_invariants import _validate_token_batch


def compare_attention_backends(
    model: GPTLanguageModel,
    tokens: torch.Tensor,
    targets: torch.Tensor,
    *,
    atol: float = 2e-5,
    rtol: float = 1e-5,
) -> dict:
    """Check logits, loss and named gradients after copying identical weights.

    Work only on temporary CPU models, leaving the caller's original
    weights, gradients, mode and global RNG untouched.
    """
    if model.config.attention_impl not in ("fused", "sdpa"):
        raise ValueError("Attention parity requires fused or sdpa GPT")
    _validate_token_batch(model, tokens)
    if next(model.parameters()).device.type != "cpu" or tokens.device.type != "cpu":
        raise ValueError("Attention backend comparison currently requires CPU")
    if (not isinstance(targets, torch.Tensor)
            or targets.shape != tokens.shape or targets.dtype != torch.long):
        raise ValueError("targets must be torch.long with shape [batch, time]")
    if not bool(((targets == -1) |
                 ((targets >= 0) & (targets < model.config.vocab_size))).all()):
        raise ValueError("Targets must be vocabulary IDs or -1")
    if not bool((targets != -1).any()):
        raise ValueError("At least one supervised target is required")

    prior_rng = torch.get_rng_state().clone()
    try:
        manual_config = replace(model.config, attention_impl="fused")
        sdpa_config = replace(model.config, attention_impl="sdpa")
        manual = GPTLanguageModel(manual_config)
        sdpa = GPTLanguageModel(sdpa_config)
        manual.load_state_dict(model.state_dict(), strict=True)
        sdpa.load_state_dict(model.state_dict(), strict=True)
        manual.eval()
        sdpa.eval()
        with torch.enable_grad():
            manual_logits, manual_loss = manual.forward_nanogpt(tokens, targets)
            sdpa_logits, sdpa_loss = sdpa.forward_nanogpt(tokens, targets)
            try:
                torch.testing.assert_close(manual_logits, sdpa_logits, atol=atol, rtol=rtol)
                torch.testing.assert_close(manual_loss, sdpa_loss, atol=atol, rtol=rtol)
            except AssertionError as exc:
                raise AssertionError("Manual and SDPA forward outputs differ") from exc
            manual_loss.backward()
            sdpa_loss.backward()

        manual_grad = dict(manual.named_parameters())
        sdpa_grad = dict(sdpa.named_parameters())
        if set(manual_grad) != set(sdpa_grad):
            raise AssertionError("Manual and SDPA gradient parameter keys differ")
        max_gradient_error = 0.0
        for name in sorted(manual_grad):
            a = manual_grad[name].grad
            b = sdpa_grad[name].grad
            if a is None or b is None:
                raise AssertionError(f"Missing gradient for {name}")
            try:
                torch.testing.assert_close(a, b, atol=atol, rtol=rtol)
            except AssertionError as exc:
                raise AssertionError(f"Manual and SDPA gradients differ at {name}") from exc
            max_gradient_error = max(
                max_gradient_error, float((a - b).abs().max())
            )
        return {
            "matched_parameter_gradients": len(manual_grad),
            "max_logit_error": float((manual_logits - sdpa_logits).abs().max()),
            "loss_error": float((manual_loss - sdpa_loss).abs()),
            "max_gradient_error": max_gradient_error,
        }
    finally:
        torch.set_rng_state(prior_rng)
