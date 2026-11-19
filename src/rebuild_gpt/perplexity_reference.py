"""Interpret next-token cross entropy relative to a uniform vocabulary baseline."""

import math

import torch
from torch.nn import functional as F


def compare_uniform_perplexity(logits: torch.Tensor, targets: torch.Tensor) -> dict:
    """Report CE in nats, bits/token, and perplexity on supervised positions.

    For a uniform vocabulary of V tokens, CE = ln(V) and perplexity = V.
    Loss alone does not establish that a model generalizes or is calibrated.
    """
    if (not isinstance(logits, torch.Tensor) or logits.ndim != 3
            or logits.numel() == 0 or not logits.is_floating_point()
            or logits.size(-1) < 2 or not bool(torch.isfinite(logits).all())):
        raise ValueError("logits must be finite floating-point [batch,time,vocab>=2]")
    if (not isinstance(targets, torch.Tensor) or targets.dtype != torch.long
            or targets.shape != logits.shape[:2]):
        raise ValueError("targets must be torch.long with shape [batch,time]")
    valid = (targets == -1) | ((targets >= 0) & (targets < logits.size(-1)))
    if not bool(valid.all()) or not bool((targets != -1).any()):
        raise ValueError("targets must include supervised vocabulary IDs or -1")
    loss = F.cross_entropy(logits.reshape(-1, logits.size(-1)),
                           targets.reshape(-1), ignore_index=-1)
    nats = float(loss.detach())
    if not math.isfinite(nats) or nats > 700:
        raise ValueError("Cross-entropy cannot be represented as finite perplexity")
    uniform_nats = math.log(logits.size(-1))
    return {
        "supervised_positions": int((targets != -1).sum()),
        "ignored_positions": int((targets == -1).sum()),
        "vocab_size": logits.size(-1),
        "cross_entropy_nats": nats,
        "bits_per_token": nats / math.log(2),
        "perplexity": math.exp(nats),
        "uniform_baseline_nats": uniform_nats,
        "uniform_baseline_perplexity": float(logits.size(-1)),
        "excess_loss_vs_uniform_nats": nats - uniform_nats,
    }
