"""Verify that adding one constant to every vocabulary logit changes nothing."""

import torch
from torch.nn import functional as F


def check_logit_shift_invariance(
    logits: torch.Tensor,
    targets: torch.Tensor,
    *,
    offset: float = 2.0,
    atol: float = 2e-6,
) -> dict:
    """Compare probabilities, masked loss and gradients under a common shift.

    Softmax is invariant to adding the same constant to each class logit
    at a position. A different shift per vocabulary class would NOT be.
    """
    if logits.ndim != 3 or logits.numel() == 0 or not logits.is_floating_point():
        raise ValueError("logits must be floating-point [batch,time,vocab]")
    if not bool(torch.isfinite(logits).all()):
        raise ValueError("logits must be finite")
    if targets.shape != logits.shape[:2] or targets.dtype != torch.long:
        raise ValueError("targets must be torch.long [batch,time]")
    if not bool(((targets == -1) |
                 ((targets >= 0) & (targets < logits.size(-1)))).all()):
        raise ValueError("targets must be vocabulary IDs or -1")
    if not bool((targets != -1).any()):
        raise ValueError("At least one target must be supervised")
    if type(offset) not in (float, int) or not torch.isfinite(torch.tensor(float(offset))):
        raise ValueError("offset must be finite")

    x = logits.detach().clone().requires_grad_(True)
    y = logits.detach().clone().requires_grad_(True)
    shift = torch.full_like(y[..., :1], float(offset))
    loss_a = F.cross_entropy(x.reshape(-1, x.size(-1)),
                             targets.reshape(-1), ignore_index=-1)
    loss_b = F.cross_entropy((y + shift).reshape(-1, y.size(-1)),
                             targets.reshape(-1), ignore_index=-1)
    gradient_a, = torch.autograd.grad(loss_a, x)
    gradient_b, = torch.autograd.grad(loss_b, y)
    probability_a = torch.softmax(x.detach(), dim=-1)
    probability_b = torch.softmax(y.detach() + shift, dim=-1)
    try:
        torch.testing.assert_close(probability_a, probability_b, atol=atol, rtol=0)
        torch.testing.assert_close(loss_a.detach(), loss_b.detach(), atol=atol, rtol=0)
        torch.testing.assert_close(gradient_a, gradient_b, atol=atol, rtol=0)
    except AssertionError as exc:
        raise AssertionError("Common logit shift changed model probabilities or loss") from exc
    return {
        "status": "invariant",
        "offset": float(offset),
        "supervised_tokens": int((targets != -1).sum()),
        "probability_error": float((probability_a - probability_b).abs().max()),
        "loss_error": float((loss_a - loss_b).detach().abs()),
        "gradient_error": float((gradient_a - gradient_b).abs().max()),
    }
